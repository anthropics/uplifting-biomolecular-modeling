"""ef2_fused_ln — a bandwidth-bound dx-only LayerNorm backward for the frozen-weight fold trunk under ``ef2_autograd_kernels``
(K-A2 / K-D3: five LayerNorm backwards per PairUpdateBlock backward, 240 per design step), residual-link gradient folded in,
BITWISE-IDENTICAL to the kernel it replaces.

Why.  In fast / big every trunk LayerNorm backward goes through ``agk._ln_bwd_dx_only``, which launches the fork's ``_ln_bwd_kernel``:
64 rows x 256 fp32 values per program on 8 warps, and the per-tile dgamma/dbeta partial sums still computed and stored (never reduced or
used: gamma/beta are frozen).  Measured on H100 at 450 tokens (M = 202 500 rows, d = 256, bf16): 0.290 ms with the residual link /
0.235 ms without = 1.4 / 1.3 TB/s on a ~3 TB/s part.  ``_ln_bwd_rows_kernel`` here does the same fp32 arithmetic in the same order
(x_hat = (x - mean) * rstd; wdy = dy * w; c1 = sum(wdy)/d; c2 = sum(wdy * x_hat)/d; dx = (wdy - (c1 + x_hat * c2)) * rstd [+ link grad])
on 8 rows per 4-warp program with no partial sums: 0.163 / 0.131 ms (2.5 / 2.4 TB/s), and its dx is tensor-equal to the vendored kernel's
(the per-row reduction tree is the same; k/test_ef2_fused_ln.py asserts it on odd shapes, both dtypes, with and without the link).
The channel-major operand of the second TriMul LayerNorm ('dbij->bijd': x is the [d, M] contraction output; 2 of the 5 calls, 0.41 ms
each = 0.75 TB/s on the vendored kernel) is, by default, LEFT on the vendored kernel, so the mode's digits do not change at all.  With
``enable(model, channel_major=True)`` it is served by ``_ln_bwd_cmajor_kernel`` (256-row m-runs x 32-column d-chunks, two pipelined sweeps):
0.26 ms (1.6x); its two row sums accumulate per d-chunk, so its dx is NOT bitwise to the vendored kernel's — max |delta| = 1 bf16 ulp,
rel-RMS 7e-6 at 450 tokens (bench_ln.py), i.e. fp32 reassociation only.

Numerics class: ``channel_major=False`` (default) — identical to the lever it accelerates (bitwise; no new deviation, fast and big as
they are); ``channel_major=True`` — fast ('reordered accumulation' of two fp32 row sums; same class as K-D3's LayerNorm statistics).

Usage — the patch point is ``ef2_autograd_kernels._ln_bwd_dx_only`` (the helper both of agk's frozen-LayerNorm backwards call:
LNFrozen for the TriMul LNs, TransitionRefround for the transition LN), so the lever is process-wide for every agk-enabled model;
idempotent and reversible; enable order relative to ``agk.enable`` is free (the helper is looked up at call time):
    import ef2_fused_ln
    ef2_fused_ln.enable(model)                        # bitwise lever; `model` accepted for the kit's lever signature (marked), helper patched
    ef2_fused_ln.enable(model, channel_major=True)    # + the channel-major kernel (fast class)
    ...
    ef2_fused_ln.disable(model)                       # restores agk's own helper object
Install it BEFORE any CUDA-graph capture of the trunk backward (a trunk graph pool captures whatever helper is bound at capture time).
``ln_bwd_dx(...)`` has agk's helper signature.  Routing is counted, never silent: ``stats()`` / ``describe()`` report served (row-major),
served_cmajor, routed_cmajor (channel-major -> vendored kernel when channel_major=False) and fallback[reason] (non-CUDA, unsupported d).
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["enable", "disable", "describe", "stats", "installed", "ln_bwd_dx", "KERNELS_OK"]

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _ln_bwd_rows_kernel(
        gy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, gr_ptr, gx_ptr, M,
        D: tl.constexpr, TILE_M: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    ):
        """row-major x / grad_x ([M, D]): TILE_M full rows per program."""
        rows = tl.program_id(0).to(tl.int64) * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        rmask = rows < M
        cols = tl.arange(0, D).to(tl.int64)
        off = rows[:, None] * D + cols[None, :]
        gy = tl.load(gy_ptr + off, mask=rmask[:, None], other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=rmask[:, None], other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=rmask, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
        w = tl.load(w_ptr + cols).to(tl.float32)
        x_hat = (x - mean[:, None]) * rstd[:, None]
        wdy = gy * w[None, :]
        c1 = tl.sum(wdy, axis=1) / D
        c2 = tl.sum(wdy * x_hat, axis=1) / D
        gx = (wdy - (c1[:, None] + x_hat * c2[:, None])) * rstd[:, None]
        if HAS_RESIDUAL:
            gx = gx + tl.load(gr_ptr + off, mask=rmask[:, None], other=0.0).to(tl.float32)
        tl.store(gx_ptr + off, gx.to(gx_ptr.type.element_ty), mask=rmask[:, None])

    @triton.jit
    def _ln_bwd_cmajor_kernel(
        gy_ptr, x_ptr, w_ptr, mean_ptr, rstd_ptr, gr_ptr, gx_ptr, M,
        D: tl.constexpr, RUN: tl.constexpr, BD: tl.constexpr, HAS_RESIDUAL: tl.constexpr, NUM_STAGES: tl.constexpr,
    ):
        """channel-major x / grad_x ([D, M]; grad_y and the link grad are [M, D]): RUN rows (contiguous m) per program, d walked in
        BD-wide chunks twice — sums, then dx — with software-pipelined chunk loads; x / grad_x traffic runs along m (512-byte runs),
        grad_y along d; the second sweep re-reads the program's own tile from L2.  The two row sums accumulate per d-chunk (fp32), a
        different association than the vendored kernel's single tree: NOT bitwise to it (<= 1 bf16 ulp; see the module docstring)."""
        rows = tl.program_id(0).to(tl.int64) * RUN + tl.arange(0, RUN).to(tl.int64)
        rmask = rows < M
        M64 = M.to(tl.int64)
        dd = tl.arange(0, BD).to(tl.int64)
        mean = tl.load(mean_ptr + rows, mask=rmask, other=0.0)
        rstd = tl.load(rstd_ptr + rows, mask=rmask, other=0.0)
        s1 = tl.zeros((RUN,), dtype=tl.float32)
        s2 = tl.zeros((RUN,), dtype=tl.float32)
        for d0 in tl.range(0, D, BD, num_stages=NUM_STAGES):
            w = tl.load(w_ptr + d0 + dd).to(tl.float32)
            gy = tl.load(gy_ptr + rows[:, None] * D + (d0 + dd)[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
            x = tl.load(x_ptr + (d0 + dd)[None, :] * M64 + rows[:, None], mask=rmask[:, None], other=0.0).to(tl.float32)
            wdy = gy * w[None, :]
            s1 += tl.sum(wdy, axis=1)
            s2 += tl.sum(wdy * ((x - mean[:, None]) * rstd[:, None]), axis=1)
        c1 = s1 / D
        c2 = s2 / D
        for d0 in tl.range(0, D, BD, num_stages=NUM_STAGES):
            w = tl.load(w_ptr + d0 + dd).to(tl.float32)
            gy = tl.load(gy_ptr + rows[:, None] * D + (d0 + dd)[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
            x = tl.load(x_ptr + (d0 + dd)[None, :] * M64 + rows[:, None], mask=rmask[:, None], other=0.0).to(tl.float32)
            x_hat = (x - mean[:, None]) * rstd[:, None]
            gx = (gy * w[None, :] - (c1[:, None] + x_hat * c2[:, None])) * rstd[:, None]
            if HAS_RESIDUAL:
                gx = gx + tl.load(gr_ptr + rows[:, None] * D + (d0 + dd)[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
            tl.store(gx_ptr + (d0 + dd)[None, :] * M64 + rows[:, None], gx.to(gx_ptr.type.element_ty), mask=rmask[:, None])

    KERNELS_OK = True
    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    KERNELS_OK = False
    _IMPORT_ERR = _e

# Fixed launch configuration (H100; D = 256 rows of bf16/fp32): memory-bound kernels, chosen for full-width coalesced segments.
ROWS_TILE_M = 8        # row-major kernel: rows per program
ROWS_NUM_WARPS = 4
CM_RUN = 256           # channel-major kernel: rows (contiguous m) per program — 256 x 2 B = 512-byte runs per d
CM_BD = 32             # ... d columns per chunk (grad_y / link-grad rows are read in 64-byte segments)
CM_NUM_WARPS = 8
CM_STAGES = 3          # ... chunk-load pipelining depth (3 stages of fp32 chunks = 192 KiB of shared memory: fits sm_90's 227 KiB opt-in)
CM_STAGES_LOWSMEM = 2  # ... on a card whose opt-in shared memory per block is below 200 KiB (sm_80: 163 KiB) with 4-byte operands: 2 stages (128 KiB); 2-byte operands keep CM_STAGES everywhere (96 KiB)


def _cm_stages(t: torch.Tensor) -> int:
    """The channel-major kernel's pipelining depth for operand tensor `t` on its device: CM_STAGES unless the operands are 4-byte and the card's
    opt-in shared memory cannot hold three chunks of them (pipelining depth changes no arithmetic: the result is the same tensor either way)."""
    if t.element_size() <= 2:
        return CM_STAGES
    p = torch.cuda.get_device_properties(t.device)
    optin = getattr(p, "shared_memory_per_block_optin", None) or (232448 if (p.major, p.minor) >= (9, 0) else 166912)
    return CM_STAGES if optin >= 200 * 1024 else CM_STAGES_LOWSMEM

LAYOUT_ROWMAJOR = 0    # the fork's _LAYOUT_BND_BND ('bijd->bijd')
LAYOUT_CMAJOR = 1      # the fork's _LAYOUT_DBN_BND ('dbij->bijd')

_STATS = {"served": 0, "served_cmajor": 0, "routed_cmajor": 0, "fallback": {}}
_SERVE_CMAJOR = False  # set by enable(channel_major=...)


def _fallback_reason(grad_y: Tensor, x_view: Tensor, D: int, layout_int: int) -> str | None:
    if not KERNELS_OK:
        return "kernels_unavailable"
    if not (grad_y.is_cuda and x_view.is_cuda):
        return "not_cuda"
    if D < 32 or D > 1024 or (D & (D - 1)) != 0:
        return f"d_{D}"
    if layout_int not in (LAYOUT_ROWMAJOR, LAYOUT_CMAJOR):
        return f"layout_{layout_int}"
    return None


_orig_helper = None    # agk's own _ln_bwd_dx_only, captured by enable(); serves the channel-major layout and any non-qualifying call


def ln_bwd_dx(grad_y: Tensor, x_view: Tensor, w: Tensor, mean: Tensor, rstd: Tensor, layout_int: int,
              grad_residual: Tensor | None, M: int, D: int) -> Tensor:
    """dx of LayerNorm(x; w, b) given dy (= agk._ln_bwd_dx_only's contract): x_view / dx are [M, D] (layout 0) or [D, M] (layout 1),
    grad_y and grad_residual are [M, D]; mean / rstd fp32 [M]; dx has x_view's dtype and layout; fp32 arithmetic."""
    reason = _fallback_reason(grad_y, x_view, D, layout_int)
    if reason is not None or (layout_int == LAYOUT_CMAJOR and not _SERVE_CMAJOR):
        if reason is not None:
            _STATS["fallback"][reason] = _STATS["fallback"].get(reason, 0) + 1
        else:
            _STATS["routed_cmajor"] += 1                                   # channel-major operand left on the vendored kernel (bitwise lever)
        if _orig_helper is None:
            raise RuntimeError(f"ef2_fused_ln.ln_bwd_dx: call not served here ({reason or 'channel-major layout'}) and enable() was not run")
        return _orig_helper(grad_y, x_view, w, mean, rstd, layout_int, grad_residual, M, D)
    grad_y = grad_y.contiguous()
    has_res = grad_residual is not None
    gr = grad_residual.contiguous() if has_res else grad_y            # dummy pointer when absent (never read: HAS_RESIDUAL constexpr)
    if layout_int == LAYOUT_ROWMAJOR:
        gx = torch.empty((M, D), device=x_view.device, dtype=x_view.dtype)
        _ln_bwd_rows_kernel[(triton.cdiv(M, ROWS_TILE_M),)](
            grad_y, x_view, w, mean, rstd, gr, gx, M, D=D, TILE_M=ROWS_TILE_M, HAS_RESIDUAL=has_res, num_warps=ROWS_NUM_WARPS)
        _STATS["served"] += 1
    else:
        gx = torch.empty((D, M), device=x_view.device, dtype=x_view.dtype)
        _ln_bwd_cmajor_kernel[(triton.cdiv(M, CM_RUN),)](
            grad_y, x_view, w, mean, rstd, gr, gx, M, D=D, RUN=CM_RUN, BD=min(CM_BD, D), HAS_RESIDUAL=has_res, NUM_STAGES=_cm_stages(x_view),
            num_warps=CM_NUM_WARPS)
        _STATS["served_cmajor"] += 1
    return gx


def enable(model: torch.nn.Module | None = None, channel_major: bool = False):
    """Route ef2_autograd_kernels' frozen-LayerNorm backward (K-A2 TriMul LNs, K-D3 transition LN) through these kernels.  The patch
    point is the module-level helper ``agk._ln_bwd_dx_only`` — process-wide for every agk-enabled model; idempotent; ``disable`` restores
    the original function object.  ``channel_major=False`` (default): row-major operands only, BITWISE to the vendored kernel, the
    channel-major TriMul LN routed to the vendored kernel; ``channel_major=True``: also serve the channel-major operand with the two-kernel
    (m-run x d-chunk) path — fp32 reassociation of the two row sums (fast class: not bitwise to the vendored digits).  Returns stats."""
    if not KERNELS_OK:
        raise RuntimeError(f"ef2_fused_ln: Triton kernels unavailable: {_IMPORT_ERR!r}")
    global _orig_helper, _SERVE_CMAJOR
    _SERVE_CMAJOR = bool(channel_major)
    import ef2_autograd_kernels as agk
    if agk._ln_bwd_dx_only is not ln_bwd_dx:
        _orig_helper = agk._ln_bwd_dx_only
        agk._ln_bwd_dx_only = ln_bwd_dx
    if model is not None:
        model._fused_ln_enabled = True
    return _STATS


def disable(model: torch.nn.Module | None = None) -> None:
    global _orig_helper, _SERVE_CMAJOR
    _SERVE_CMAJOR = False
    import ef2_autograd_kernels as agk
    if agk._ln_bwd_dx_only is ln_bwd_dx and _orig_helper is not None:
        agk._ln_bwd_dx_only = _orig_helper
    _orig_helper = None
    if model is not None and hasattr(model, "_fused_ln_enabled"):
        del model._fused_ln_enabled


def installed() -> bool:
    import ef2_autograd_kernels as agk
    return agk._ln_bwd_dx_only is ln_bwd_dx


def stats() -> dict:
    return {"served": _STATS["served"], "served_cmajor": _STATS["served_cmajor"], "routed_cmajor": _STATS["routed_cmajor"],
            "fallback": dict(_STATS["fallback"]), "installed": installed(), "channel_major": _SERVE_CMAJOR}


def describe(model: torch.nn.Module | None = None) -> str:
    s = stats()
    fb = ",".join(f"{k}={v}" for k, v in sorted(s["fallback"].items())) or "none"
    return (f"fused_ln: installed={s['installed']} channel_major={s['channel_major']} served={s['served']} served_cmajor={s['served_cmajor']} "
            f"routed_cmajor={s['routed_cmajor']} fallback={fb} rows_tile={ROWS_TILE_M}x{ROWS_NUM_WARPS}w cmajor_tile={CM_RUN}x{CM_BD}x{CM_NUM_WARPS}w x{CM_STAGES}")
