"""protenix_fpf_dit_attn_exact — lever `dit_attn_exact` (class EXACT-BITWISE): the DiffusionTransformer's pair-bias attention
(protenix.model.modules.primitives._attention -> torch.nn.functional.scaled_dot_product_attention, fp32, head dim 48, bias broadcast over
the diffusion samples) served by a prebuilt sm_90 CUDA kernel whose output is bit-identical to the memory-efficient SDPA kernel it replaces.

apply()   gate PTX_DIT_ATTN_EXACT=1; loads prebuilt/<torch>-cu<cuda>-sm<cc>/dit_attn_exact.so, checks the manifest (torch, cuda, arch,
          sha256 of the .so and of csrc/), runs the load-time load-time check (3 real-shaped cases incl. a partial key block and a padded-pitch bias view:
          torch.equal against torch's SDPA in this process + output digests against the manifest) and installs the route; returns the marker
          "DITATTN:on(...)"; raises RuntimeError("dit_attn_exact: <reason>") when it cannot run here (the mode refuses by name).
report()  {"unit": "dit_attn_exact", "calls": n, "routes": {"kernel": n, "<reason>": n, ...}} — calls outside the envelope take the original
          statement and are counted by reason (declared route, decided from shapes/strides before the call).
Envelope: use_efficient_implementation=True, attn_bias given; after the stock fp32 upcast: q,k,v CUDA fp32 [B,H,N,48] (same N for q and k/v),
last-dim contiguous, row strides multiple of 4 floats, 16-byte aligned; bias fp32 [1|B,H,N,N] with last-dim stride 1 (any row pitch).
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "dit_attn_exact"
STATS: Dict[str, Any] = {"calls": 0, "routes": {}, "installed": False, "marker": None, "loadcheck": None}
_MOD = None
_ORIG = None


def _sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def source_sha256() -> Dict[str, str]:
    d = os.path.join(HERE, "csrc")
    return {fn: _sha_file(os.path.join(d, fn)) for fn in sorted(os.listdir(d)) if fn.endswith((".cu", ".cuh", ".h"))}


def stack_key() -> str:
    cc = torch.cuda.get_device_capability()
    return f"torch{torch.__version__.split('+')[0]}-cu{(torch.version.cuda or 'none').replace('.', '')}-sm{cc[0]}{cc[1]}"


def prebuilt_dir(key: Optional[str] = None) -> str:
    return os.path.join(HERE, "prebuilt", key or stack_key())


def loadcheck_cases(device="cuda"):
    """Deterministic real-shaped cases (CPU generator -> device): (q,k,v,bias) in the model's layout: [B,N,H*D] linear outputs viewed
    [B,N,H,D].transpose(1,2), q / sqrt(48); bias [1,H,N,N] fp32 (case 3: a view with an 8-aligned row pitch, as a cached bias may be)."""
    import math
    out = []
    for i, (B, H, N, pitch8) in enumerate(((5, 16, 200, False), (1, 16, 64, False), (2, 16, 100, True))):
        g = torch.Generator(device="cpu"); g.manual_seed(4800 + i)
        def lin():
            return torch.randn(B, N, H * 48, generator=g, dtype=torch.float32).to(device).view(B, N, H, 48).transpose(1, 2)
        q = lin() / math.sqrt(48); k = lin(); v = lin()
        b = torch.randn(1, H, N, N, generator=g, dtype=torch.float32).to(device)
        if pitch8:
            P = (N + 7) // 8 * 8
            buf = torch.zeros(1, H, N, P, dtype=torch.float32, device=device); buf[..., :N] = b; b = buf[..., :N]
        out.append((q, k, v, b))
    return out


def run_loadcheck(mod, manifest: Optional[dict] = None) -> Dict[str, Any]:
    import torch.nn.functional as F
    rep = {"cases": 0, "bit_equal_vs_sdpa": True, "digests": [], "digest_match": None}
    with torch.no_grad():
        for (q, k, v, b) in loadcheck_cases():
            ref = F.scaled_dot_product_attention(q, k, v, attn_mask=b, scale=1.0)
            out = mod.forward(q, k, v, b, 1)
            torch.cuda.synchronize()
            rep["cases"] += 1
            rep["bit_equal_vs_sdpa"] = rep["bit_equal_vs_sdpa"] and bool(torch.equal(out, ref))
            rep["digests"].append(_sha_tensor(out))
    if manifest is not None and manifest.get("loadcheck_digests"):
        rep["digest_match"] = rep["digests"] == manifest["loadcheck_digests"]
    return rep


def load_prebuilt(key: Optional[str] = None):
    """Load and check the prebuilt extension for this stack; raises RuntimeError('dit_attn_exact: ...') by name on any failed check."""
    global _MOD
    if _MOD is not None:
        return _MOD
    if not torch.cuda.is_available():
        raise RuntimeError(f"{NAME}: no CUDA device")
    cc = torch.cuda.get_device_capability()
    pdir = prebuilt_dir(key)
    mpath = os.path.join(pdir, "manifest.json")
    if not os.path.isfile(mpath):
        raise RuntimeError(f"{NAME}: no prebuilt for stack {stack_key()} (expected {mpath})")
    man = json.load(open(mpath))
    if f"sm{cc[0]}{cc[1]}" not in man.get("arch", []):
        raise RuntimeError(f"{NAME}: device cc {cc} not in the prebuilt arch list {man.get('arch')}")
    if man.get("torch") != torch.__version__ or man.get("cuda") != torch.version.cuda:
        raise RuntimeError(f"{NAME}: prebuilt built for torch {man.get('torch')} / cuda {man.get('cuda')}, running torch {torch.__version__} / cuda {torch.version.cuda}")
    so = os.path.join(pdir, man["so"])
    from opt_core.gates import binary_refusal  # noqa: PLC0415
    why = binary_refusal(so)
    if why:
        raise RuntimeError(f"{NAME}: {man['so']} refused: {why}")
    if _sha_file(so) != man["so_sha256"]:
        raise RuntimeError(f"{NAME}: sha256 of {man['so']} does not match the manifest")
    if source_sha256() != man["source_sha256"]:
        raise RuntimeError(f"{NAME}: csrc/ sha256 does not match the manifest (the .so was built from different sources)")
    loader = importlib.machinery.ExtensionFileLoader(man["module_name"], so)
    spec = importlib.util.spec_from_file_location(man["module_name"], so, loader=loader)
    mod = importlib.util.module_from_spec(spec); loader.exec_module(mod)
    if str(getattr(mod, "version", "")) != str(man.get("kernel_version")):
        raise RuntimeError(f"{NAME}: loaded module version {getattr(mod, 'version', None)} != manifest {man.get('kernel_version')}")
    mod.init()
    sc = run_loadcheck(mod, man)
    STATS["loadcheck"] = sc
    if not sc["bit_equal_vs_sdpa"]:
        raise RuntimeError(f"{NAME}: load-time load-time check: output not bit-identical to torch SDPA in this process ({sc})")
    if sc["digest_match"] is False:
        raise RuntimeError(f"{NAME}: load-time load-time check: output digests differ from the manifest ({sc})")
    mod._dit_manifest = man
    _MOD = mod
    return mod


def _route(q, k, v, b) -> Optional[str]:
    """None = kernel; else the reason token for the original statement (decided from dtypes/shapes/strides only)."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or b.dim() != 4:
        return "rank"
    if q.shape[-1] != 48:
        return "head_dim"
    if not (q.is_cuda and q.dtype == torch.float32 and k.dtype == torch.float32 and v.dtype == torch.float32 and b.dtype == torch.float32):
        return "dtype"
    B, H, N, D = q.shape
    if tuple(k.shape) != (B, H, N, D) or tuple(v.shape) != (B, H, N, D):
        return "kv_shape"
    if b.shape[0] not in (1, B) or tuple(b.shape[1:]) != (H, N, N):
        return "bias_shape"
    for tsr in (q, k, v):
        if tsr.stride(3) != 1 or tsr.stride(2) % 4 or tsr.data_ptr() % 16:
            return "layout"
    if q.stride(0) % 2 or q.stride(1) % 2 or k.stride(0) % 4 or k.stride(1) % 4 or v.stride(0) % 4 or v.stride(1) % 4:
        return "layout"
    if b.stride(3) != 1:
        return "bias_layout"
    if B * H * ((N + 127) // 128) >= (1 << 30):
        return "size"
    return None


def _count(route: str):
    STATS["calls"] += 1
    STATS["routes"][route] = STATS["routes"].get(route, 0) + 1


def apply() -> str:
    """Install the route on protenix.model.modules.primitives._attention. Returns the marker; raises by name when the lever cannot run."""
    global _ORIG
    if os.environ.get("PTX_DIT_ATTN_EXACT", "0") in ("", "0"):
        return "DITATTN:inactive(PTX_DIT_ATTN_EXACT unset)"
    if STATS["installed"]:
        return STATS["marker"]
    mod = load_prebuilt()
    import protenix.model.modules.primitives as PR
    _ORIG = PR._attention

    def _attention(q, k, v, attn_bias=None, use_efficient_implementation=True, inplace_safe=False):
        if use_efficient_implementation and attn_bias is not None:
            q32 = q.to(dtype=torch.float32); k32 = k.to(dtype=torch.float32); b32 = attn_bias.to(dtype=torch.float32)   # the stock statement's own upcasts
            why = _route(q32, k32, v, b32)
            if why is None:
                _count("kernel")
                return mod.forward(q32, k32, v, b32, 1)
            _count(why)
        else:
            _count("not_sdpa" if not use_efficient_implementation else "no_bias")
        return _ORIG(q, k, v, attn_bias=attn_bias, use_efficient_implementation=use_efficient_implementation, inplace_safe=inplace_safe)

    _attention._dit_attn_exact = True
    PR._attention = _attention
    man = mod._dit_manifest
    sc = STATS["loadcheck"]
    STATS["installed"] = True
    STATS["marker"] = (f"DITATTN:on(protenix_fpf_dit_attn_exact v{man.get('kernel_version')} {os.path.basename(os.path.dirname(mod.__file__))} "
                       f"loadcheck={sc['cases']}cases-bit-equal digests={'match' if sc['digest_match'] else 'n/a'}; EXACT-BITWISE vs mem-efficient SDPA fp32 D=48)")
    return STATS["marker"]


def report() -> Dict[str, Any]:
    return {"unit": NAME, "pid": os.getpid(), "calls": STATS["calls"], "routes": dict(STATS["routes"]), "installed": STATS["installed"],
            "loadcheck": STATS["loadcheck"]}
