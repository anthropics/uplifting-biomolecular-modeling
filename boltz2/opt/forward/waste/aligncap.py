"""aligncap.py — for the sampler's roll-out graph (bz_sampler.py): the 3x3 SVD of Boltz-2's per-step weighted_rigid_align as a unit that is BITWISE
torch.linalg.svd(cov32, driver="gesvd") (what boltz 2.2.1's loss/diffusionv2.py:51 calls on CUDA) but issues NO torch-level host sync
(torch's `_linalg_check_errors(info)` device->host read is gone; devInfo stays on the device and is folded into a flag the caller reads once
per item), writes into caller-owned static buffers, and allocates its cuSOLVER workspace once (init()).

How it is bitwise: torch 2.12's CUDA `linalg.svd(..., driver="gesvd")` (ATen BatchLinearAlgebraLib.cpp svd_cusolver -> svd_cusolver_gesvd ->
apply_svd_cusolver_gesvd) calls cusolverDn<S>gesvd once PER MATRIX of the batch on a column-major clone of A with jobu = jobvt = 'A',
lda = ldu = ldvt = 3, the queried lwork and an rwork of min(m,n) floats, U written F-contiguous, VT written into an n x n workspace that is then
copied into the (C-contiguous) Vh output. This unit makes the identical library call (same routine, same job/ld parameters, same input bytes,
handle bound to torch's current stream) through ctypes on the libcusolver torch itself loaded, so U, S, VT are the same bytes; U is written
straight into an F-contiguous [Bm,3,3] view (strides (9,1,3), exactly torch's), Vh[i] = copy of the VT workspace read transposed (C-contig,
strides (9,3,1), torch's), S strides (3,1). Bit-compared on random / covariance-like / rank-deficient / repeated-singular-value / zero
matrices (torch.equal on (U,S,Vh)), and at every step of 200- to 1400-token predictions inside the kit
worker (factors AND the final aligned coordinates equal the stock function's; output files sha256 = stock).

WHAT IT CANNOT DO (measured): cusolverDnSgesvd is a hybrid host/device routine — it BLOCKS the calling host thread
until the stream has drained, and it is NOT legal under
stream capture (cudaErrorStreamCaptureInvalidated — and a torch.linalg.svd attempted under capture poisons cuSOLVER for
the process: never probe it inside a worker). There is no other routine that returns the same bits (gesvdj/gesvdjBatched are Jacobi = different
numerics = fast tier only). Consequence for the exact tier: the SVD stays an eager call BETWEEN captured segments; the recoverable part of the
per-step eager glue is everything else (centroids/covariance before it; rotation/det/apply + Euler update after it), which `align_pre` /
`align_post` below expose as the stock statements split at the SVD so the sampler roll-out can capture them (torch.det is capturable).

API (the sampler roll-out's contract):
    init(Bm, device)                      -> allocate handle/workspace/flags once, OUTSIDE capture (eager step 0); idempotent per (Bm, device)
    aligncap(cov32, U_out, S_out, Vh_out) -> None; cov32 [Bm,3,3] fp32 contiguous; U_out = new_U(Bm) (F-contig view, strides (9,1,3)),
                                             S_out [Bm,3] contiguous, Vh_out [Bm,3,3] contiguous; values bitwise torch.linalg.svd(cov32, driver='gesvd')
    new_U(Bm, device) / new_S / new_Vh    -> correctly-strided static output buffers
    flags() -> {"gesvd_info_nonzero": bool, "calls": int}  (ONE host read; call once per item, after the loop) ; reset_flags()
    align_pre(true_coords, pred_coords, weights, mask) / align_post(...)  -> stock weighted_rigid_align split at the SVD (verbatim statements;
                                             the two print-only guards returned as device bools for the caller to accumulate and read once per item)
"""
from __future__ import annotations

import ctypes
import os
from typing import Dict, Optional

import torch

PIN_TORCH = "2.12"          # torch major.minor whose ATen gesvd wrapper (per-matrix cusolverDnSgesvd, jobu=jobvt='A', column-major clone, VT workspace copy) was matched
PIN_CUSOLVER = 12004        # cusolverGetVersion() of the pinned stack's libcusolver (nvidia cu13 wheel)

_S: Dict[str, object] = {"L": None, "h": None, "lib": None, "lwork": None, "WORK": None, "RWORK": None, "INFO": None, "VT": None, "Bm": 0,
                         "device": None, "calls": 0}


def _load() -> None:
    if _S["L"] is not None:
        return
    torch.linalg.svd(torch.eye(3, device="cuda"))          # makes torch load (and initialise) its libcusolver; we bind THAT library
    libs = sorted({l.split()[-1] for l in open(f"/proc/{os.getpid()}/maps").read().splitlines() if "libcusolver" in l})
    if not libs:
        raise RuntimeError("[aligncap] libcusolver not mapped after torch.linalg.svd — cannot bind the routine torch uses")
    L = ctypes.CDLL(libs[0])
    L.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    L.cusolverDnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    L.cusolverDnSgesvd_bufferSize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    L.cusolverDnSgesvd.argtypes = [ctypes.c_void_p, ctypes.c_byte, ctypes.c_byte, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_void_p]
    # pin: the bitwise claim holds for torch 2.12 + libcusolver 12004 (the pinned CUDA 13.0 stack); off-pin -> refuse by name
    L.cusolverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    ver = ctypes.c_int(0); L.cusolverGetVersion(ctypes.byref(ver))
    tv = torch.__version__.split("+")[0]
    if not tv.startswith(PIN_TORCH) or ver.value != PIN_CUSOLVER:
        raise RuntimeError(f"[aligncap] NOT ACTIVE: reason=pin_mismatch torch={tv} (pinned {PIN_TORCH}.x) libcusolver={ver.value} (pinned {PIN_CUSOLVER}) — "
                           "re-establish the bitwise proof (waste_svd_probe.py + tests/test_aligncap.py) on this stack before re-pinning")
    h = ctypes.c_void_p()
    st = L.cusolverDnCreate(ctypes.byref(h))
    if st != 0:
        raise RuntimeError(f"[aligncap] cusolverDnCreate status {st}")
    lw = ctypes.c_int(0)
    st = L.cusolverDnSgesvd_bufferSize(h, 3, 3, ctypes.byref(lw))
    if st != 0 or lw.value < 0:
        raise RuntimeError(f"[aligncap] cusolverDnSgesvd_bufferSize status {st} lwork {lw.value}")
    _S.update(L=L, h=h, lib=libs[0], lwork=lw.value, cusolver_version=ver.value)


def init(Bm: int, device="cuda") -> None:
    """Allocate the cuSOLVER handle, the gesvd workspace (lwork floats + rwork), the devInfo/flag block and the VT staging buffer ONCE, outside any capture."""
    _load()
    dev = torch.device(device)
    if _S["WORK"] is not None and _S["Bm"] >= Bm and _S["device"] == dev:
        return
    _S["WORK"] = torch.empty(max(int(_S["lwork"]), 1), dtype=torch.float32, device=dev)
    _S["RWORK"] = torch.empty(3, dtype=torch.float32, device=dev)
    _S["INFO"] = torch.zeros(max(Bm, 1), dtype=torch.int32, device=dev)
    _S["VT"] = torch.empty(3, 3, dtype=torch.float32, device=dev)
    _S["Bm"] = Bm; _S["device"] = dev


def new_U(Bm: int, device="cuda") -> torch.Tensor:
    """[Bm,3,3] fp32 with torch.linalg.svd's U strides (9,1,3) (F-contiguous per matrix) — gesvd writes U column-major straight into it."""
    return torch.empty(Bm, 3, 3, dtype=torch.float32, device=device).mT


def new_S(Bm: int, device="cuda") -> torch.Tensor:
    return torch.empty(Bm, 3, dtype=torch.float32, device=device)


def new_Vh(Bm: int, device="cuda") -> torch.Tensor:
    return torch.empty(Bm, 3, 3, dtype=torch.float32, device=device)


def aligncap(cov32: torch.Tensor, U_out: torch.Tensor, S_out: torch.Tensor, Vh_out: torch.Tensor) -> None:
    """U_out, S_out, Vh_out <- SVD of each cov32[i] (values bitwise torch.linalg.svd(cov32, driver='gesvd')); no torch-level host sync.
    NOTE: the cuSOLVER routine itself blocks the host until the current stream drains and is not stream-capturable (see module doc)."""
    if torch.cuda.is_current_stream_capturing():      # the routine is hybrid host/device — never inside a capture (it invalidates the capture and poisons cuSOLVER)
        raise RuntimeError("[aligncap] called under CUDA stream capture: cusolverDnSgesvd is not capturable — place the SVD between captured segments")
    Bm = cov32.shape[0]
    if cov32.dtype != torch.float32 or cov32.shape[1:] != (3, 3) or not cov32.is_contiguous() or not cov32.is_cuda:
        raise ValueError(f"[aligncap] cov32 must be a contiguous CUDA fp32 [Bm,3,3] tensor, got {cov32.dtype} {tuple(cov32.shape)} contiguous={cov32.is_contiguous()}")
    if U_out.shape != (Bm, 3, 3) or U_out.stride() != (9, 1, 3) or S_out.shape != (Bm, 3) or not S_out.is_contiguous() or Vh_out.shape != (Bm, 3, 3) or not Vh_out.is_contiguous():
        raise ValueError("[aligncap] output buffers: U_out [Bm,3,3] strides (9,1,3) (use new_U), S_out [Bm,3] contiguous, Vh_out [Bm,3,3] contiguous")
    if _S["WORK"] is None or _S["Bm"] < Bm or _S["device"] != cov32.device:
        init(Bm, cov32.device)
    L, h = _S["L"], _S["h"]
    Af = cov32.mT.contiguous()          # per matrix: row-major A^T == column-major A (ATen: cloneBatchedColumnMajor); gesvd overwrites it
    VT, WORK, RWORK, INFO = _S["VT"], _S["WORK"], _S["RWORK"], _S["INFO"]
    L.cusolverDnSetStream(h, ctypes.c_void_p(torch.cuda.current_stream(cov32.device).cuda_stream))
    for i in range(Bm):                 # ATen loops the batch with one gesvd per matrix and a shared VT workspace; so do we
        st = L.cusolverDnSgesvd(h, 65, 65, 3, 3, ctypes.c_void_p(Af[i].data_ptr()), 3, ctypes.c_void_p(S_out[i].data_ptr()),
                                ctypes.c_void_p(U_out[i].data_ptr()), 3, ctypes.c_void_p(VT.data_ptr()), 3, ctypes.c_void_p(WORK.data_ptr()), int(_S["lwork"]),
                                ctypes.c_void_p(RWORK.data_ptr()), ctypes.c_void_p(INFO[i:i + 1].data_ptr()))
        if st != 0:
            raise RuntimeError(f"[aligncap] cusolverDnSgesvd status {st}")
        Vh_out[i].copy_(VT.mT)          # gesvd wrote VT column-major into VT's storage; torch's Vh[i] (C-contig) == that storage read transposed
    _S["calls"] = int(_S["calls"]) + Bm


def svd(cov32: torch.Tensor):
    """Convenience: allocate correctly-strided outputs and run aligncap; returns (U, S, Vh) like torch.linalg.svd(cov32, driver='gesvd')."""
    Bm = cov32.shape[0]
    U, S, Vh = new_U(Bm, cov32.device), new_S(Bm, cov32.device), new_Vh(Bm, cov32.device)
    aligncap(cov32, U, S, Vh)
    return U, S, Vh


def flags() -> Dict[str, object]:
    """ONE host read: whether any gesvd call since the last reset reported devInfo != 0 (stock: torch raises at that step). Call once per item."""
    if _S["INFO"] is None:
        return {"gesvd_info_nonzero": False, "calls": 0}
    return {"gesvd_info_nonzero": bool((_S["INFO"] != 0).any().item()), "calls": int(_S["calls"])}


def reset_flags() -> None:
    if _S["INFO"] is not None:
        _S["INFO"].zero_()
    _S["calls"] = 0


# ------------------------------------------------------------------------------------------ stock weighted_rigid_align, split at the SVD
def align_pre(true_coords, pred_coords, weights, mask):
    """boltz 2.2.1 loss/diffusionv2.py weighted_rigid_align, statements BEFORE the SVD, verbatim; returns the state the second half needs plus the
    first print-guard's condition as a DEVICE bool (stock host-syncs on it to print a warning)."""
    from einops import einsum
    out_shape = torch.broadcast_shapes(true_coords.shape, pred_coords.shape)
    *batch_size, num_points, dim = out_shape
    weights = (mask * weights).unsqueeze(-1)
    true_centroid = (true_coords * weights).sum(dim=-2, keepdim=True) / weights.sum(dim=-2, keepdim=True)
    pred_centroid = (pred_coords * weights).sum(dim=-2, keepdim=True) / weights.sum(dim=-2, keepdim=True)
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid
    guard_small_cloud = torch.any(mask.sum(dim=-1) < (dim + 1))       # stock: if <this>: print("Warning: The size of one of the point clouds is <= dim+1. ...")
    cov_matrix = einsum(weights * pred_coords_centered, true_coords_centered, "... n i, ... n j -> ... i j")
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    return {"cov32": cov_matrix_32, "true_centered": true_coords_centered, "pred_centroid": pred_centroid, "batch_size": batch_size,
            "num_points": num_points, "dim": dim, "original_dtype": original_dtype, "device": cov_matrix.device, "guard_small_cloud": guard_small_cloud}


def align_post(pre: dict, U, S, V):
    """statements AFTER the SVD, verbatim (U, S, V = what torch.linalg.svd returned, i.e. V is Vh here exactly as in stock before `V = V.mH`);
    returns (aligned_coords, guard_low_rank as a DEVICE bool)."""
    from einops import einsum
    batch_size, num_points, dim = pre["batch_size"], pre["num_points"], pre["dim"]
    V = V.mH
    guard_low_rank = (S.abs() <= 1e-15).any() & (not (num_points < (dim + 1)))   # stock: if <this>: print("Warning: Excessively low rank of ...")
    rot_matrix = torch.einsum("... i j, ... k j -> ... i k", U, V).to(dtype=torch.float32)
    F = torch.eye(dim, dtype=pre["cov32"].dtype, device=pre["device"])[None].repeat(*batch_size, 1, 1)
    F[..., -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "... i j, ... j k, ... l k -> ... i l")
    rot_matrix = rot_matrix.to(dtype=pre["original_dtype"])
    aligned_coords = einsum(pre["true_centered"], rot_matrix, "... n i, ... j i -> ... n j") + pre["pred_centroid"]
    aligned_coords.detach_()
    return aligned_coords, guard_low_rank


def weighted_rigid_align_nosync(true_coords, pred_coords, weights, mask, guards: Optional[torch.Tensor] = None):
    """Drop-in for the stock function: same outputs bit for bit, zero torch-level host syncs (the SVD still blocks inside cuSOLVER).
    `guards` (optional bool[2] device tensor) accumulates the two print conditions for one deferred read per item."""
    pre = align_pre(true_coords, pred_coords, weights, mask)
    cov = pre["cov32"]; shp = cov.shape
    U, S_, Vh = svd(cov.reshape(-1, 3, 3).contiguous())
    if U.dim() != len(shp):             # extra leading batch dims only (the sampler's cov is always [Bm,3,3]: nothing is reshaped there)
        U = U.reshape(shp); S_ = S_.reshape(shp[:-1]); Vh = Vh.reshape(shp)
    out, g2 = align_post(pre, U, S_, Vh)
    if guards is not None:
        guards[0] |= pre["guard_small_cloud"]; guards[1] |= g2
    return out
