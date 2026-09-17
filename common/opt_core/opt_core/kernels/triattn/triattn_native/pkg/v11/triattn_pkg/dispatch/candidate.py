"""`best` -- the routed triangle-attention forward: one entry point that sends each call to the fastest measured kernel of this tree
for its (compute capability, sequence length, head dim, dtype, mask, memory layout).

    --kernel best=candidate:best[:bf16bias]        (harness plug-in contract: fn(q, k, v, bias, mask=None, scale=None) -> out)

Routing table.  The device's compute capability selects the table; the measured cells behind every row are the package's CELLS.json
(one record per harness case, keyed by cc); the route names are the keys of PINS.

  cc 9.0 (H100), bf16, D = 32, S_q == S_kv (any N incl. N != S, rank-4 operands, fp32 or 16-bit bias, masked or not incl. per-row padding):
      S < 512, contiguous  -> k13     (dispatch/kernels/k13.py: persistent warp-specialized Gluon kernel, mask through the tensor core)
      S < 512, strided     -> cuda    (cuda/triattn_cuda.py:forward: warp-specialized wgmma / TMA kernel)
      512 <= S <= 3072     -> cuda_b  (cuda_b/triattn_m1.py:triangle_attention_m1: three consumer warpgroups, max-free streaming softmax,
                                       exact fp32 bias staging, per-row dead key-tile skip, SAFE fix pass; contiguous or strided)
      3072 < S <= 4096     -> HIGH_BAND_KERNEL = cuda_c (cuda_c/triattn_mw.py:triangle_attention: TMA-multicast variant, geometry 3x4)
      S > 4096             -> cuda_b  (cuda_c and cuda refuse S > 4096 by name; cuda_b serves any S)
  cc 9.0, fp16, or D = 16 (S_q == S_kv):  S <= 640 -> k13;  above -> tri (triton/candidate.py:tri = the k12 Gluon kernel)
  cc 9.0, D in {64, 128}, or S_q != S_kv -> tri (its k10 Triton path)

  cc 8.0 (A100), bf16, D in {16, 32, 64}, S_q == S_kv (any S >= 1, any N incl. N != S, any H, B >= 1, rank-4 operands, fp32 or 16-bit
      bias, key masks of every kind), contiguous or strided views with stride(-1) == 1
                           -> cuda_80 (cuda_80/triattn_sm80.py:triangle_attention_sm80: mma.sync m16n8k16 + ldmatrix + cp.async
                                       multistage pipeline, the family's max-free streaming softmax and exact fp32 bias staging, SAFE fix pass)
  cc 8.0, fp16, or D = 128, or S_q != S_kv -> tri (k10; unmeasured on this device)

  any other cc (8.6, 8.9, 10.x, 12.x, ...): every servable shape -> tri (k10 / k12 by name; unmeasured) -- the CUDA members are built and
      measured for exactly one architecture each (sm_90a, sm_80) and are not routed elsewhere.
  fp32 inputs, other head dims, per-row biases, per-query masks, non-CUDA tensors -> Unsupported (typed, raised before any work);
  a Gluon route (k13, tri's k12 path) on an interpreter whose triton lacks triton.experimental.gluon -> Unsupported("needs_triton_gluon: ...")
      by name, before any work (never rerouted silently).

Every routed kernel is imported from its own directory of this tree (each directory unmodified at the commit recorded in PINS); the
router puts those directories on sys.path and imports them under their own top-level module names.
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)                       # the tree holding dispatch/ triton/ cuda/ cuda_c/ cuda_b/ cuda_80/
for p in (HERE, os.path.join(PKG_ROOT, "triton"), os.path.join(PKG_ROOT, "cuda"), os.path.join(PKG_ROOT, "cuda_c"), os.path.join(PKG_ROOT, "cuda_b"),
          os.path.join(PKG_ROOT, "cuda_80")):
    if p not in sys.path:
        sys.path.insert(0, p)

from triattn.k11 import Unsupported  # noqa: E402  (the typed refusal: a NotImplementedError naming the reason)

from pins import PINS  # noqa: E402  (routed kernel directories pinned at the commit that sealed them; verified by test_dispatch.py pins)

CUDA_B_MIN_S = 512        # cc 9.0, bf16 D=32: cuda_b from here (contiguous and strided, any mask, any bias dtype); below: k13 (contiguous) / cuda (strided)
CUDA_B_MAX_S = 3072       # ... up to here; above: the high band
HIGH_BAND_KERNEL = "cuda_c"   # cc 9.0, bf16 D=32, 3072 < S <= 4096: "cuda_c" or "cuda_b"; S > 4096 -> cuda_b (cuda_c and cuda refuse it by name)
CUDA_C_MAX_S = 4096       # cuda_c serves S <= 4096
K13_MAX_S = 640          # k13 up to here (its per-item fixed costs are amortised; k12's table path wins on masked rows above)
GLUON_DIMS = (16, 32)
K10_DIMS = (16, 32, 64, 128)
SM80_DIMS = (16, 32, 64)  # cc 8.0: head dims the sm_80 member serves (bf16, S_q == S_kv); D = 128 -> tri
SM90 = (9, 0)
SM80 = (8, 0)

_impl = {}


def _load(name):
    f = _impl.get(name)
    if f is None:
        if name == "k13":
            from kernels.k13 import triattn_k13 as f
        elif name == "tri":
            from candidate_tri import tri as f          # triton/candidate.py, imported under an unambiguous module name
        elif name == "cuda":
            from triattn_cuda import forward as f
        elif name == "cuda_c":
            from triattn_mw import triangle_attention as f      # cuda_c/triattn_mw.py
        elif name == "cuda_b":
            from triattn_m1 import triangle_attention_m1 as f   # cuda_b/triattn_m1.py (default instantiation, flags 0)
        elif name == "cuda_80":
            from triattn_sm80 import triangle_attention_sm80 as f   # cuda_80/triattn_sm80.py (the sm_80 member)
        else:
            raise KeyError(name)
        _impl[name] = f
    return f


def _import_tri_candidate():
    """triton/candidate.py defines `tri`; load it as module `candidate_tri` so it cannot shadow / be shadowed by this file."""
    import importlib.util
    if "candidate_tri" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("candidate_tri", os.path.join(PKG_ROOT, "triton", "candidate.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_tri"] = mod
    spec.loader.exec_module(mod)


_CC = {}
_GLUON = None
K12_MIN_S_16 = 256        # tri picks k12 (Gluon) from here when the bias arrives in q's dtype (triton/candidate.py K12_MIN_S_16) ...
K12_MIN_S_32 = 320        # ... and from here with an fp32 bias (K12_MIN_S_32); below, and for D 64|128 / S_q != S_kv / other devices, tri = k10 (plain Triton)


def _gluon_available() -> bool:
    """Whether this interpreter's triton carries the Gluon dialect (triton.experimental.gluon, Triton >= 3.6) the k13 / k12 routes are written in."""
    global _GLUON
    if _GLUON is None:
        import importlib.util
        try:
            _GLUON = importlib.util.find_spec("triton.experimental.gluon") is not None
        except (ImportError, ValueError):
            _GLUON = False
    return _GLUON


def _gluon_route(name: str) -> str:
    """`name` (a Gluon route) if Gluon is importable here, else the typed refusal -- raised before any work, never rerouted silently."""
    if not _gluon_available():
        raise Unsupported(f"needs_triton_gluon: route {name} is a Gluon kernel (triton.experimental.gluon, Triton >= 3.6) and this interpreter's triton does not carry it")
    return name


def _cc(dev) -> tuple:
    """(major, minor) compute capability of `dev`, memoised per device (the routing table is keyed by it)."""
    r = _CC.get(dev)
    if r is None:
        r = tuple(torch.cuda.get_device_capability(dev))
        _CC[dev] = r
    return r


def _sm90(dev) -> bool:
    return _cc(dev) == SM90


def route(q, k, v, bias, mask=None) -> str:
    """The kernel `best` uses for these arguments (name in PINS), or raises Unsupported naming the reason."""
    if not (torch.is_tensor(q) and q.is_cuda):
        raise Unsupported("CUDA tensors only")
    if q.dim() == 4:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    if q.dim() != 5 or k.dim() != 5 or v.dim() != 5:
        raise Unsupported(f"unsupported rank: q/k/v must be [B,N,H,S,D] or [N,H,S,D], got {q.dim()}/{k.dim()}/{v.dim()}")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise Unsupported(f"unsupported dtype {q.dtype}/{k.dtype}/{v.dtype} (bf16 / fp16 only)")
    D, S_q, S_k = q.shape[-1], q.shape[-2], k.shape[-2]
    B, N, H = q.shape[0], q.shape[1], q.shape[2]
    if D not in K10_DIMS:
        raise Unsupported(f"unsupported head_dim {D} (supported: {K10_DIMS})")
    if tuple(k.shape) != (B, N, H, S_k, D) or tuple(v.shape) != (B, N, H, S_k, D):
        raise Unsupported(f"unsupported shape: k/v must be [B,N,H,S_kv,D] = {[B, N, H, S_k, D]}, got {list(k.shape)} / {list(v.shape)}")
    if bias is not None:
        bb = bias if bias.dim() == 5 else (bias.unsqueeze(0) if bias.dim() == 4 else bias)
        if bb.dim() != 5 or tuple(bb.shape) != (B, 1, H, S_q, S_k):
            raise Unsupported(f"unsupported bias shape {list(bias.shape)}: must be [B,1,H,S_q,S_kv] = {[B, 1, H, S_q, S_k]} (one pair bias shared by all N rows; per-row biases are not triangle attention)")
    if mask is not None:
        mm = mask if mask.dim() == 5 else (mask.unsqueeze(0) if mask.dim() == 4 else mask)
        if mm.dim() != 5 or tuple(mm.shape) != (B, N, 1, 1, S_k):
            raise Unsupported(f"unsupported mask shape {list(mask.shape)}: must be a key mask [B,N,1,1,S_kv] = {[B, N, 1, 1, S_k]} (per-query masks are not supported)")
    dense = q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    cc = _cc(q.device)
    sm90 = cc == SM90
    bf16_d32 = sm90 and q.dtype == torch.bfloat16 and D == 32 and S_q == S_k
    gluon_ok = sm90 and S_q == S_k and D in GLUON_DIMS
    if bf16_d32:
        if S_q < CUDA_B_MIN_S:
            return _gluon_route("k13") if dense else "cuda"                # S < 512: k13 on contiguous operands, cuda on strided views
        if S_q <= CUDA_B_MAX_S or S_q > CUDA_C_MAX_S or HIGH_BAND_KERNEL == "cuda_b":
            if os.environ.get("TRIATTN_M1_FLAGS") or os.environ.get("TRIATTN_M1_PLUG_FLAGS"):
                raise Unsupported("TRIATTN_M1_FLAGS / TRIATTN_M1_PLUG_FLAGS is set: best routes only cuda_b's reviewed default instantiation (flags 0)")
            return "cuda_b"                                                # 512 <= S <= 3072, and S > 4096 (cuda_c / cuda refuse it by name)
        if os.environ.get("MW_GEOM"):
            raise Unsupported("MW_GEOM is set: best routes only cuda_c's reviewed default geometry (R=3, ring depth 4)")
        return "cuda_c"                                                    # 3072 < S <= 4096 (HIGH_BAND_KERNEL)
    if cc == SM80 and q.dtype == torch.bfloat16 and D in SM80_DIMS and S_q == S_k:
        return "cuda_80"                                                   # cc 8.0: the sm_80 member, every S, contiguous or strided (stride(-1) == 1; other views are refused by name)
    if gluon_ok and S_q <= K13_MAX_S:
        return _gluon_route("k13")                                         # fp16 / D=16, small S
    if gluon_ok and S_q >= (K12_MIN_S_16 if (bias is not None and bias.dtype == q.dtype) else K12_MIN_S_32):
        return _gluon_route("tri")                                         # fp16 / D=16 above 640: tri takes its k12 (Gluon) path there
    return "tri"                                                           # D 64|128, S_q != S_kv, other devices: tri = k10 (plain Triton), by name


def best(q, k, v, bias, mask=None, scale=None):
    name = route(q, k, v, bias, mask)
    if q.dim() == 4:                                     # trunk operands without a batch dim: [N,H,S,D] -> [1,N,H,S,D] views (no copies)
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        bias = bias.unsqueeze(0) if bias.dim() == 4 else bias
        mask = mask.unsqueeze(0) if (mask is not None and mask.dim() == 4) else mask
        squeeze = True
    else:
        squeeze = False
    if name == "tri":
        _import_tri_candidate()
    f = _load(name)
    try:
        if name in ("cuda", "cuda_c", "cuda_b", "cuda_80"):
            out = f(q, k, v, bias, mask=mask, scale=scale)
        else:
            out = f(q, k, v, bias, mask, scale)
    except Unsupported:
        raise
    except NotImplementedError as e:                    # a kernel directory's own typed refusal (each defines its Unsupported): re-typed, never escapes raw
        raise Unsupported(f"{name}: {e}") from e
    return out[0] if squeeze else out
