"""gather_attn -- fused INDEX-SET attention with an additive pair bias (forward / inference), one Triton kernel (strategy ``F5.gather_attn``).

The op. Every query attends to an explicit SET of keys, listed per (batch element, query) as ``k`` key indices, with a per-pair additive
bias read from a pair tensor addressed by absolute (query, key) position and an optional elementwise output gate::

    out[b, i, h*dh:(h+1)*dh] = gate[b, i, h*dh:(h+1)*dh] * sum_{j in set(idx[b|0, i, :])} softmax_j( scale * <q[b,i,h,:], k[b,j,h,:]> + bias[b|0, i, j, h] ) v[b, j, h, :]

This is the local ("sparse", "windowed", "neighbour-list") attention of all-atom structure models: the atom transformers of the
AlphaFold3 family give each atom query a fixed number of keys — a neighbour list from a distance top-k plus sequence neighbours
(one engine's ``create_attention_indices``: k = 128 atom keys, 32 token keys) or a sequence window (32 queries x 128 keys per block:
a window IS an index set — every query of a block lists the block's 128 keys, batch-shared, so ``idx`` has batch extent 1). The two
stock formulations of the op materialize either a dense ``[B, H, LQ, LK]`` bias-plus-(-inf)-mask tensor for a fused SDPA call, or
``[B, LQ, k, H*dh]`` gathered copies of K / V; this kernel materializes neither: per query it gathers the k rows of K and V (L2-resident)
and the k bias scalars straight from the bias tensor in the layout the model produces it in (``[B|1, LQ, LK, H]``, any strides; batch
extent 1 = one bias shared by the batch). Work and traffic are O(B * LQ * k * H*dh), independent of LK and of where the listed keys are.
NO allocation grows with LQ x LK: the only allocation is the output ``[B, LQ, H*dh]`` (``stats()['max_transient_bytes']`` records it).

Index-set conventions. ``idx`` is ``[B|1, LQ, k]`` int32, rows sorted ascending (``ensure_sorted=True``, the default, checks each call
and sorts rows that are not — counted in ``stats()['rows_sorted_here']``; pass ``ensure_sorted=False`` when the producer sorts).
SET semantics: an index listed twice in a row counts once (adjacent duplicates of a sorted row are masked in-kernel), so a producer may
pad a short row by repeating an index; ``-1`` is padding too (masked). A row whose set is empty yields NaN, as the dense formulation does.
Keys need not be distinct from queries, LQ and LK are independent, k need not divide anything.

Numerics = the dense SDPA formulation's rounding points: q and k rounded to v.dtype before the product (16-bit inputs pass unchanged),
products, bias add, running max and sum in fp32, the probabilities rounded to v.dtype before P.V, fp32 accumulation, output in v.dtype.
Only the summation order over keys differs from a fused SDPA kernel (chunks of KC listed keys, ascending): a tolerance-tier replacement
of the dense path, never bit-exact. Deterministic run to run: fixed loop order, no atomics, no split-k, launch geometry a fixed function
of the head-dim class and the device's compute capability (``launch_config``: the capability's ``ROWS_BY_CC`` geometry, else ``CONFIGS``;
changing (KC, BQ, warps) changes the summation order, so they are never tuned at run time).

Cells. ``CELLS`` lists the (head dim, k, v.dtype) cells measured against the fp32 reference and the dense SDPA path on H100 (status
``certified``); ``ROWS_BY_CC`` holds the cells and geometry of the capabilities measured on their own card (``"<cc>|*"`` the
capability's row, ``"<cc>|<triton major.minor>"`` a named exception for one environment: ``opt_core.kernels.safe_settings``): a device whose
capability has a row is served exactly that row's certified cells and geometry (``cells_key``); a capability without a row (cc 9.0, the card
``CELLS`` / ``CONFIGS`` were measured on) is served those. Any other cell inside the kernel's envelope (dh <= 128, any k, bf16 / fp16) is
``candidate`` and is served only when the caller opts in (``allow_candidate=True`` or ``OPT_CORE_GATHER_ALLOW_CANDIDATE=1``, a
qualification-job switch) — otherwise refused BY NAME.
Refusals (``supported(...)`` returns the reason, ``gather_attn(...)`` raises ``Refusal`` with it; nothing falls back silently):
``no-triton`` (Triton not importable in this interpreter), ``not-cuda``, ``v-dtype-<t>``, ``qk-dtype-<t>``, ``idx-not-int32``,
``layout`` (q/k/v not ``[B, L, H*dh]``), ``head-dim-<dh>`` (> 128), ``non-unit-last-stride``, ``bias-layout``, ``bias-shape``,
``idx-shape``, ``gate-shape``, ``grad`` (autograd live on an input), ``cell-uncertified:dh<d>/k<k>/<dtype>+off(not-measured)`` (a cell inside the
envelope that no engine measured: the caller's dense path BY NAME — opt_core.kernels.cell_words — unless the candidate opt-in serves it).

    from opt_core.kernels import gather_attn as GA          # or a kit routes the name:  opt_core.kernels.route("gather_attn"); import gather_attn
    why = GA.supported(q, k, v, bias, idx, n_head)          # None = servable
    out = GA.gather_attn(q, k, v, bias, idx, n_head, gate=g)   # [B, LQ, H*dh] in v.dtype
    ref = GA.reference(q, k, v, bias, idx, n_head, gate=g)     # fp32 dense-masked reference of the same op (any device)

Shapes: q ``[B, LQ, H*dh]`` (fp32 or 16-bit), k / v ``[B, LK, H*dh]`` (k in q's dtype, v 16-bit), bias ``[B|1, LQ, LK, H]`` (any strides,
any float dtype), idx ``[B|1, LQ, k]`` int32, gate ``[B, LQ, H*dh]`` or None, out ``[B, LQ, H*dh]`` (v.dtype; ``out=`` to supply it).
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch

from opt_core.kernels import safe_settings as _safe     # the core's ONE cc|triton row resolution (stdlib at import)
from opt_core.kernels import cell_words as _cw               # the measured-off qualifier of the cell refusal word

try:                                                     # Triton ships with CUDA torch wheels; a CPU interpreter imports this module without it
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                      # pragma: no cover — absence is a named refusal at call time ('no-triton'), never a reroute
    triton = None
    tl = None
    HAVE_TRITON = False

__all__ = ["gather_attn", "supported", "reference", "Refusal", "CELLS", "CONFIGS", "ROWS_BY_CC", "cell_status", "cells_key", "launch_config",
           "candidate_allowed", "stats", "KERNEL_VERSION", "ALLOW_CANDIDATE_ENV", "HAVE_TRITON"]

KERNEL_VERSION = "gather_attn/1.0"
STRATEGY = "F5.gather_attn"
ALLOW_CANDIDATE_ENV = "OPT_CORE_GATHER_ALLOW_CANDIDATE"

# (head dim, k, v dtype) cells measured on H100 (torch 2.13.0+cu130 / triton 3.7.1) against the fp32 reference and the dense cuDNN SDPA path:
# max|d| vs fp32 <= 1.1e-2, rel-RMS 3.2e-3 at unit-scale inputs (the dense path: 1.2e-2 / 3.6e-3), run-to-run bit-exact, B 8..256, L 1.4k..7k / 100..500.
CELLS = {
    (32, 128, "bfloat16"): "certified",     # all-atom atom transformers: 4 heads x 32, 128 listed keys per atom
    (48, 32, "bfloat16"): "certified",      # token diffusion transformers: 16 heads x 48, 32 listed keys per token
}
# launch geometry by head-dim class: dh <= class -> (KC listed keys per step, BQ queries per program, num_warps). Fixed: part of the numerics.
CONFIGS = {32: (8, 16, 4), 64: (4, 16, 4), 128: (4, 16, 4)}
# Rows by compute capability — "<cc>|<triton major.minor>" (a NAMED EXCEPTION for one known environment) else "<cc>|*" (the capability's row),
# resolved for the device of the call by opt_core.kernels.safe_settings (row_for_device / key_for_device): ROWS_BY_CC[key]["cells"] = the
# (head dim, k, v dtype) cells certified ON THAT CAPABILITY (measured there against the fp32 reference and the dense SDPA path, run-to-run
# bit-exact; a cell absent from the row is a candidate there), ["configs"] = its launch geometry by head-dim class (a class absent from the row:
# CONFIGS). A capability without a row — cc 9.0, the card CELLS / CONFIGS were measured on — is served CELLS / CONFIGS.
#   "8.0|*"   A100-SXM4-80GB (cc 8.0), torch 2.13.0+cu130 / triton 3.7.1 and torch 2.4.0+cu121 / triton 3.0.0 (tests/gpu/test_gather_attn_gpu.py green on
#             both): geometry = CONFIGS' (its output on the card is what CONFIGS produced there). Device time per call (calls captured in one CUDA
#             graph, replayed), B 8: (32, 128, bf16), 4 heads, batch-shared bias / index sets, L 1.4k / 2.8k / 7k: 396 / 645 / 1620 us; rel-RMS vs the
#             fp32 reference 2.03e-3, max|d| 3.9e-3 (unit-scale inputs; the dense bf16 formulation: 6.8e-3 / 1.9e-2); run-to-run bit-exact.
#             (48, 32, bf16), 16 heads, per-sample bias, L 100 / 200 / 500: 56 / 120 / 270 us, no alternative consistently faster (-9 % .. +16 %
#             across L); rel-RMS 1.93e-3, max|d| 7.5e-3 (dense bf16: 5.4e-3 / 2.2e-2). Measured faster class-32 geometry, NOT bitwise-equal to
#             (8, 16, 4)'s output (KC halves the key-chunk summation; max|d| 3.9e-3 between the two, same error class vs the reference, run-to-run
#             bit-exact): (KC 4, BQ 16, 8 warps) = 318 / 521 / 1306 us (x1.24; fp32 q/k at 2.8k: 569 vs 724 us), the fastest of 27 register-resident
#             (KC, BQ, warps) at every L — the row {32: (4, 16, 8)} once the consuming fast rows are recorded with it.
ROWS_BY_CC: Dict[str, Dict[str, dict]] = {
    "8.0|*": {"cells": {(32, 128, "bfloat16"): "certified", (48, 32, "bfloat16"): "certified"},
              "configs": {32: (8, 16, 4), 64: (4, 16, 4), 128: (4, 16, 4)}},
}
_ROWS_RESOLVED: Dict[tuple, Optional[dict]] = {}     # device slot -> the ROWS_BY_CC row serving it, None = CELLS / CONFIGS (safe_settings.row_for_device memo)

_STATS: Dict[str, int] = {"calls": 0, "served_certified": 0, "served_candidate": 0, "rows_sorted_here": 0, "max_transient_bytes": 0}


class Refusal(ValueError):
    """A call the kernel does not serve, by name (``.reason``)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"gather_attn refused: {reason}" + (f" ({detail})" if detail else ""))
        self.reason = reason
        self.detail = detail


def candidate_allowed(flag: Optional[bool] = None) -> bool:
    """The candidate-cell opt-in of a call: the explicit flag, else ``OPT_CORE_GATHER_ALLOW_CANDIDATE=1``."""
    return bool(flag) if flag is not None else os.environ.get(ALLOW_CANDIDATE_ENV, "") == "1"


_NET = _safe.SafeNet("gather_attn")                    # names the default-table engagement of a capability without a row, once per process (cells_note())


def _row(device=None) -> Optional[dict]:
    """The ROWS_BY_CC row serving ``device`` (default: the current CUDA device), None when CELLS / CONFIGS serve (no CUDA, a CPU device, cc 9.0, a capability
    without a row — the last said ONCE: ``[opt_core/gather_attn] default settings (no row for cc <cc>, …); tuned rows exist for cc …``, ``cells_note() == 'default:no_row'``)."""
    row = _safe.row_for_device(ROWS_BY_CC, device, None, cache=_ROWS_RESOLVED)
    if row is None and _NET.note() is None:                                        # an UNKNOWN capability: the H100 cells and geometry serve it, named
        _cw.name_default_row(_NET, ROWS_BY_CC, _safe.device_cc(device), _safe.triton_mm())
    return row


def cells_note(device=None) -> Optional[str]:
    """``default:no_row`` once a CUDA device of a capability without a row (other than cc 9.0, whose measurements CELLS / CONFIGS are) was served the defaults, else None."""
    _row(device)
    return _NET.note()


def cells_key(device=None) -> Optional[str]:
    """The ROWS_BY_CC key serving ``device`` — ``"<cc>|<mm>"`` when that named-exception row exists, else ``"<cc>|*"`` when the capability has a row,
    else None (CELLS / CONFIGS serve)."""
    return _safe.key_for_device(ROWS_BY_CC, device)


def cell_status(dh: int, k: int, v_dtype: torch.dtype, device=None) -> str:
    """``certified`` | ``candidate`` (inside the envelope, not measured) for a (head dim, k, v dtype) cell on ``device``'s capability (default: the
    current CUDA device): the capability's row when it has one, else ``CELLS``."""
    row = _row(device)
    cells = row["cells"] if row is not None else CELLS
    return cells.get((int(dh), int(k), str(v_dtype).replace("torch.", "")), "candidate")


def launch_config(dh: int, device=None) -> Tuple[int, int, int]:
    """(KC listed keys per step, BQ queries per program, num_warps) for head dim ``dh`` on ``device``'s capability: the capability's row (a head-dim
    class it does not name: ``CONFIGS``), else ``CONFIGS``."""
    cls = _dh_class(dh)
    row = _row(device)
    return tuple((row["configs"] if row is not None else CONFIGS).get(cls, CONFIGS[cls]))


def stats() -> Dict[str, int]:
    """Process counters: calls, served_certified, served_candidate, rows_sorted_here, max_transient_bytes (largest output allocated here)."""
    return dict(_STATS)


def _dh_class(dh: int) -> int:
    return 32 if dh <= 32 else 64 if dh <= 64 else 128


def supported(q, k, v, bias, idx, n_head: int, gate=None, allow_candidate: Optional[bool] = None) -> Optional[str]:
    """None when the call is servable here, else the refusal reason (a short name)."""
    if not HAVE_TRITON:
        return "no-triton"
    if not (q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda and idx.is_cuda):
        return "not-cuda"
    if v.dtype not in (torch.bfloat16, torch.float16):
        return f"v-dtype-{v.dtype}".replace("torch.", "")
    if q.dtype not in (torch.float32, torch.bfloat16, torch.float16) or k.dtype != q.dtype:
        return f"qk-dtype-{q.dtype}".replace("torch.", "")
    if idx.dtype != torch.int32:
        return "idx-not-int32"
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or n_head < 1 or q.shape[-1] % n_head or k.shape[-1] != q.shape[-1] or v.shape != k.shape or k.shape[0] != q.shape[0]:
        return "layout"
    dh = q.shape[-1] // n_head
    if dh > 128:
        return f"head-dim-{dh}"
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1 or idx.stride(-1) != 1 or (gate is not None and gate.stride(-1) != 1):
        return "non-unit-last-stride"
    B, LQ, C = q.shape
    LK = k.shape[1]
    if bias.ndim != 4 or bias.shape[-1] != n_head:
        return "bias-layout"
    if bias.shape[0] not in (1, B) or bias.shape[1] != LQ or bias.shape[2] != LK:
        return "bias-shape"
    if idx.ndim != 3 or idx.shape[0] not in (1, B) or idx.shape[1] != LQ or idx.shape[2] < 1:
        return "idx-shape"
    if gate is not None and tuple(gate.shape) != (B, LQ, C):
        return "gate-shape"
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, bias) if t.is_floating_point()):
        return "grad"
    if cell_status(dh, idx.shape[2], v.dtype, q.device) != "certified" and not candidate_allowed(allow_candidate):
        return _cw.with_off(f"cell-uncertified:dh{dh}/k{idx.shape[2]}/{str(v.dtype).replace('torch.', '')}")     # measured off: no engine's record covers the cell
    return None


if HAVE_TRITON:

    @triton.jit
    def _gather_attn_fwd(
        Q, K, V, Bias, Idx, Gate, Out,
        sqb, sqm, skb, skn, svb, svn,
        sbb, sbi, sbj, sbh,
        sib, sim,
        sgb, sgm, sob, som,
        LQ, KTOT, qk_scale,
        HEAD_DIM: tl.constexpr, DPAD: tl.constexpr, BQ: tl.constexpr, KC: tl.constexpr,
        HAS_GATE: tl.constexpr, ROUND_QK: tl.constexpr,
    ):
        # one program = BQ queries of one (batch element, head); grid (query tiles, heads, batch): the query tiles and heads of one batch
        # element are adjacent in launch order, so its K / V rows stay L2-resident while they are gathered.
        pid_q = tl.program_id(0)
        h = tl.program_id(1)
        h64 = h.to(tl.int64)
        b64 = tl.program_id(2).to(tl.int64)
        rows = pid_q * BQ + tl.arange(0, BQ)
        row_ok = rows < LQ
        rows64 = rows.to(tl.int64)
        offs_e = tl.arange(0, DPAD)
        e_ok = offs_e < HEAD_DIM
        col0 = h * HEAD_DIM

        q = tl.load(Q + b64 * sqb + rows64[:, None] * sqm + col0 + offs_e[None, :], mask=row_ok[:, None] & e_ok[None, :], other=0.0)
        if ROUND_QK:
            q = q.to(V.dtype.element_ty)
        q = q.to(tl.float32)

        NEG: tl.constexpr = -1.0e30
        m_i = tl.zeros([BQ], dtype=tl.float32) + NEG
        l_i = tl.zeros([BQ], dtype=tl.float32)
        acc = tl.zeros([BQ, DPAD], dtype=tl.float32)

        idx_row = Idx + b64 * sib + rows64[:, None] * sim
        k_base = K + b64 * skb + col0
        v_base = V + b64 * svb + col0
        bias_row = Bias + b64 * sbb + rows64[:, None] * sbi + h64 * sbh            # int64: sbh may be LQ*LK (any bias strides)

        for kc in range(0, KTOT, KC):
            jj = kc + tl.arange(0, KC)
            j_ok = jj < KTOT
            valid = row_ok[:, None] & j_ok[None, :]
            idx = tl.load(idx_row + jj[None, :], mask=valid, other=-1)                     # [BQ, KC] int32
            prev = tl.load(idx_row + (jj[None, :] - 1), mask=valid & (jj[None, :] >= 1), other=-1)
            valid = valid & (idx >= 0) & (idx != prev)                                    # -1 padding and adjacent duplicates: masked
            idx64 = tl.where(valid, idx, 0).to(tl.int64)
            kg = tl.load(k_base + idx64[:, :, None] * skn + offs_e[None, None, :], mask=valid[:, :, None] & e_ok[None, None, :], other=0.0)
            if ROUND_QK:
                kg = kg.to(V.dtype.element_ty)
            s = tl.sum(q[:, None, :] * kg.to(tl.float32), 2) * qk_scale                   # [BQ, KC] fp32
            bg = tl.load(bias_row + idx64 * sbj, mask=valid, other=0.0).to(tl.float32)
            s = tl.where(valid, s + bg, NEG)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            p = tl.where(valid, p, 0.0)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            vg = tl.load(v_base + idx64[:, :, None] * svn + offs_e[None, None, :], mask=valid[:, :, None] & e_ok[None, None, :], other=0.0)
            pr = p.to(V.dtype.element_ty).to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(pr[:, :, None] * vg.to(tl.float32), 1)
            m_i = m_new

        out = acc / l_i[:, None]
        if HAS_GATE:
            g = tl.load(Gate + b64 * sgb + rows64[:, None] * sgm + col0 + offs_e[None, :], mask=row_ok[:, None] & e_ok[None, :], other=0.0)
            out = out * g.to(tl.float32)
        tl.store(Out + b64 * sob + rows64[:, None] * som + col0 + offs_e[None, :], out.to(Out.dtype.element_ty), mask=row_ok[:, None] & e_ok[None, :])


def gather_attn(q, k, v, bias, idx, n_head: int, gate=None, scale: Optional[float] = None, out=None, *, ensure_sorted: bool = True,
                allow_candidate: Optional[bool] = None, round_qk: bool = True):
    """The op over the listed key sets; returns ``out`` ``[B, LQ, H*dh]`` in v.dtype. Raises :class:`Refusal` (by name) for a call outside the
    served envelope or an untested cell without the opt-in; never falls back."""
    why = supported(q, k, v, bias, idx, n_head, gate=gate, allow_candidate=allow_candidate)
    if why is not None:
        raise Refusal(why, f"q{tuple(q.shape)}:{q.dtype} k{tuple(k.shape)} v:{v.dtype} bias{tuple(bias.shape)} idx{tuple(idx.shape)}:{idx.dtype} H={n_head}".replace("torch.", ""))
    B, LQ, C = q.shape
    dh = C // n_head
    KTOT = idx.shape[2]
    if ensure_sorted and KTOT > 1 and not bool((idx[..., 1:] >= idx[..., :-1]).all()):
        idx = torch.sort(idx, dim=-1).values.contiguous()
        _STATS["rows_sorted_here"] += 1
    KC, BQ, NW = launch_config(dh, q.device)
    DPAD = max(16, triton.next_power_of_2(dh))
    if out is None:
        out = torch.empty((B, LQ, C), dtype=v.dtype, device=q.device)
        _STATS["max_transient_bytes"] = max(_STATS["max_transient_bytes"], out.numel() * out.element_size())
    elif tuple(out.shape) != (B, LQ, C) or out.stride(-1) != 1 or not out.is_cuda:
        raise Refusal("out-layout", f"out{tuple(out.shape)}")
    scale = (1.0 / math.sqrt(dh)) if scale is None else float(scale)
    g = gate if gate is not None else out
    grid = (triton.cdiv(LQ, BQ), n_head, B)
    _gather_attn_fwd[grid](
        q, k, v, bias, idx, g, out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        0 if bias.shape[0] == 1 else bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
        0 if idx.shape[0] == 1 else idx.stride(0), idx.stride(1),
        g.stride(0), g.stride(1), out.stride(0), out.stride(1),
        LQ, KTOT, scale,
        HEAD_DIM=dh, DPAD=DPAD, BQ=BQ, KC=KC, HAS_GATE=gate is not None, ROUND_QK=round_qk,
        num_warps=NW, num_stages=2,
    )
    _STATS["calls"] += 1
    _STATS["served_certified" if cell_status(dh, KTOT, v.dtype, q.device) == "certified" else "served_candidate"] += 1
    return out


def reference(q, k, v, bias, idx, n_head: int, gate=None, scale: Optional[float] = None, dtype=torch.float32, round_qk: bool = False):
    """The op as the dense formulation computes it, in ``dtype`` (fp32 by default), on any device: scatter the sets into a ``[B, LQ, LK]`` mask
    (``-1`` entries and duplicates are set members once / not at all), ``softmax(scale q.k + bias)`` masked to -inf outside the set, times v,
    times gate. ``round_qk=True`` rounds q, k to v.dtype first (the served kernel's rounding point). Materializes ``[B, H, LQ, LK]``: a reference,
    not a serving path."""
    B, LQ, C = q.shape
    LK = k.shape[1]
    dh = C // n_head
    scale = (1.0 / math.sqrt(dh)) if scale is None else float(scale)
    ib = idx.shape[0]
    valid = torch.zeros((ib, LQ, LK + 1), dtype=torch.bool, device=q.device)          # column LK absorbs the -1 padding
    valid.scatter_(2, torch.where(idx < 0, LK, idx).to(torch.int64), True)
    valid = valid[..., :LK].expand(B, -1, -1)
    if round_qk:
        q, k = q.to(v.dtype), k.to(v.dtype)
    qh = q.to(dtype).reshape(B, LQ, n_head, dh).transpose(1, 2)
    kh = k.to(dtype).reshape(B, LK, n_head, dh).transpose(1, 2)
    vh = v.to(dtype).reshape(B, LK, n_head, dh).transpose(1, 2)
    s = qh @ kh.transpose(-1, -2) * scale + bias.to(dtype).permute(0, 3, 1, 2)
    s = s.masked_fill(~valid[:, None], float("-inf"))
    o = (torch.softmax(s, -1) @ vh).transpose(1, 2).reshape(B, LQ, C)
    if gate is not None:
        o = o * gate.to(dtype)
    return o.contiguous()
