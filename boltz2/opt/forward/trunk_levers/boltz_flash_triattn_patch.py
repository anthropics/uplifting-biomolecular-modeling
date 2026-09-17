"""
boltz_flash_triattn_patch.py -- runtime patch for Boltz-2 (boltz 2.2.1): route TriangleAttention (starting + ending node; trunk pairformer,
MSA-module pair stack, template pairformer, confidence pairformer -- they all use the same class) to the Triton flash triangle-attention
kernel (flash_triattn.py, used as is) when the environment variable BOLTZ_TRIATTN=flash is set.

Default (BOLTZ_TRIATTN unset / anything else): NOTHING is patched -> stock bytes, stock numerics.

Two entry paths are covered:
  (b) the --no_kernels path (the program's frozen reference invocation): TriangleAttention.forward is replaced by a function that, when the
      call is eligible (CUDA, bf16/fp16 autocast or bf16 input, no grad, head_dim in {16,32,64,128}, N >= BOLTZ_TRIATTN_MIN_TOKENS), computes
      layer_norm -> triangle bias -> UNscaled q,k,v (Attention._prep_qkv(apply_scale=False)) -> flash(q,k,v,bias,mask.bool(),scale=1/sqrt(c_hidden))
      -> gating/linear_o (Attention._wrap_up), bypassing chunk_layer entirely (flash never materialises the [H,N,N] logits, so the 128-row
      query chunking of the stock path is unnecessary); otherwise it calls the exact stock forward (stock result for that call).
  (a) the kernels-on path: boltz.model.layers.triangular_attention.primitives.kernel_triangular_attn (the cuEquivariance call) is replaced by
      a function with the same gates that calls flash instead of cuequivariance_torch.triangle_attention (per-call fallback to cuEq).
      (With the class patch active, (a) is only reached by calls that (b) already sent down the stock road; it applies the same decision.)

Numerics of the flash path: bf16 tensor-core products, fp32 accumulation, fp32 logits (scale*q.k + bias), exact fp32 online softmax, P cast to
bf16 for P.V, no atomics, tile config a fixed function of head_dim -> bitwise reproducible run-to-run.  This is the cuEq/FlashAttention scheme;
it is NOT bitwise the stock torch path (which rounds the logits to bf16 and takes the softmax in bf16) -> Tier-2 lever, compare before enabling.

Usage (driver process, before or after model construction):
    import boltz_flash_triattn_patch as F2P; F2P.apply()          # returns True iff the patch is active
    ... run boltz ...
    print(F2P.report())                                             # frozen settings + per-call-site counters

Env knobs (read ONCE at apply(); nothing is re-read per call):
    BOLTZ_TRIATTN=flash              enable
    BOLTZ_TRIATTN_WARM=1|"n1,n2,.." AOT-compile the Triton kernel for those token counts at apply() time (see warm_shapes()).
    BOLTZ_TRIATTN_MIN_TOKENS=<n>     size gate: calls with fewer than n tokens (pair-rep rows) use the stock path (default %(DEFAULT_MIN_TOKENS)s)
    BOLTZ_TRIATTN_EXACT=0            folded-log2e exp2 variant of the kernel (default unset/1 = exact exp)
    BOLTZ_TRIATTN_REPORT=1           print report() as one JSON line at interpreter exit
    BOLTZ_TRIATTN_{CONFIG,BIAS16,BIAS16_MIN_SK,FASTLAUNCH,LIBDEVICE}  forwarded to flash_triattn's PTX_TRIATTN_* knobs (same meaning) unless
                                     the PTX_TRIATTN_* variable is already set.
Preflight:  BOLTZ_TRIATTN=flash python boltz_flash_triattn_patch.py --preflight   -> prints 'PREFLIGHT F2 APPLIED ...' and exits 0 on success.
"""
from __future__ import annotations

import atexit
import collections
import hashlib
import json
import math
import os
import sys
import threading
from typing import Any, Dict, Optional, Tuple

import torch
from opt_core.oom import is_oom          # the core's one out-of-memory predicate: an out-of-memory error propagates, no fallback applied

DEFAULT_MIN_TOKENS = 0          # size gate default (tokens): no lower gate is needed; raise it to shrink the covered surface if desired
                                # (the mode rows set BOLTZ_TRIATTN_MIN_TOKENS themselves).
__doc__ = __doc__ % {"DEFAULT_MIN_TOKENS": DEFAULT_MIN_TOKENS}
PATCH_VERSION = "1.1-warm+f2-boltz-v1"

_LOCK = threading.Lock()
STATS: "collections.Counter[str]" = collections.Counter()
_LABELS: Dict[int, str] = {}
_STATE: Dict[str, Any] = {"applied": False, "mode": None, "orig_forward": None, "orig_kernel_fn": None, "orig_boltz2_forward": None,
                          "exact_exp": None, "use_libdevice": None, "min_tokens": DEFAULT_MIN_TOKENS,
                          "flash_sha256": None, "flash_file": None, "forwarded_env": {}, "errors": []}
_FWD_KNOBS = ("CONFIG", "BIAS16", "BIAS16_MIN_SK", "FASTLAUNCH", "LIBDEVICE", "EXACT")


def _forward_env_knobs() -> Dict[str, str]:
    """BOLTZ_TRIATTN_<X> -> PTX_TRIATTN_<X> (flash_triattn.py reads the PTX_ names once at import). Runs BEFORE flash_triattn is imported."""
    fwd = {}
    for k in _FWD_KNOBS:
        v = os.environ.get("BOLTZ_TRIATTN_" + k)
        if v is not None and os.environ.get("PTX_TRIATTN_" + k) is None:
            os.environ["PTX_TRIATTN_" + k] = v
            fwd[k] = v
    return fwd


def _import_flash():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import flash_triattn as FT  # noqa: E402
    if _STATE["flash_sha256"] is None:
        try:
            _STATE["flash_file"] = FT.__file__
            with open(FT.__file__, "rb") as fh:
                _STATE["flash_sha256"] = hashlib.sha256(fh.read()).hexdigest()
        except Exception:
            pass
    return FT


def _autocast_dtype() -> Optional[torch.dtype]:
    if not torch.is_autocast_enabled():
        return None
    try:
        return torch.get_autocast_dtype("cuda")
    except Exception:  # older torch
        return torch.get_autocast_gpu_dtype()


def label_modules(model: torch.nn.Module) -> int:
    """Give every TriangleAttention submodule of `model` a call-site label '<top-level component>.<start|end>' used in the counters."""
    from boltz.model.layers.triangular_attention.attention import TriangleAttention
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, TriangleAttention):
            top = name.split(".")[0] if name else "model"
            _LABELS[id(m)] = f"{top}.{'start' if m.starting else 'end'}"
            n += 1
    return n


def _site(self) -> str:
    return _LABELS.get(id(self)) or f"{type(self).__name__}.{'start' if getattr(self, 'starting', True) else 'end'}"


def _eligible(self, x: torch.Tensor) -> Tuple[bool, str]:
    if not x.is_cuda:
        return False, "not_cuda"
    if x.dim() != 4:
        return False, f"rank{x.dim()}"
    n = int(x.shape[-2])
    if n < _STATE["min_tokens"]:
        return False, f"n<{_STATE['min_tokens']}"
    dt = _autocast_dtype() or x.dtype
    if dt not in (torch.bfloat16, torch.float16):
        return False, f"dtype_{str(dt).replace('torch.', '')}"
    d = getattr(self.mha, "c_hidden", None)
    if d not in (16, 32, 64, 128):
        return False, f"head_dim_{d}"
    if torch.is_grad_enabled() and x.requires_grad:
        return False, "requires_grad"      # forward-only kernel
    return True, "ok"


def _flash_core(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
    """Flash replacement of TriangleAttention.forward's body: the stock preamble verbatim, then unscaled q,k,v -> flash -> gate/linear_o.
    chunk_layer is bypassed (flash is O(N) in memory)."""
    from boltz.model.layers.triangular_attention.utils import permute_final_dims
    FT = _import_flash()
    if mask is None:
        mask = x.new_ones(x.shape[:-1])
    if not self.starting:
        x = x.transpose(-2, -3)
        mask = mask.transpose(-1, -2)
    x = self.layer_norm(x)                                              # [B, I, J, C]
    mask5 = mask[..., :, None, None, :]                                 # [B, I, 1, 1, J]
    tri = permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)   # [B, 1, H, I, J]
    mha = self.mha
    q, k, v = mha._prep_qkv(x, x, apply_scale=False)                    # [B, I, H, J, D] transposed views, UNscaled (as the kernels-on path)
    o = FT.flash_triangle_attention(q, k, v, tri, mask=mask5.bool(), scale=1.0 / math.sqrt(mha.c_hidden),
                                     exact_exp=_STATE["exact_exp"], use_libdevice=_STATE["use_libdevice"])   # [B, I, H, J, D]
    o = o.transpose(-2, -3)                                             # [B, I, J, H, D]
    o = mha._wrap_up(o, x)                                              # sigmoid gate + linear_o with q_x = layer-normed x (as stock)
    if not self.starting:
        o = o.transpose(-2, -3)
    return o, tuple(q.shape), str(q.dtype).replace("torch.", "")


def _patched_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, chunk_size: Optional[int] = None, use_kernels: bool = False):
    orig = _STATE["orig_forward"]
    STATS["calls"] += 1
    ok, why = _eligible(self, x)
    site = _site(self)
    n = int(x.shape[-2]) if x.dim() >= 2 else -1
    if not ok:
        STATS[f"stock|{why}|{site}|N={n}|use_kernels={bool(use_kernels)}"] += 1
        STATS["stock_calls"] += 1
        return orig(self, x, mask=mask, chunk_size=chunk_size, use_kernels=use_kernels)
    try:
        out, qshape, qdt = _flash_core(self, x, mask)
    except Exception as e:  # safety net: exact stock result for this call — except a GPU out-of-memory error, which propagates (no fallback on out-of-memory)
        if is_oom(e): raise
        key = "stock|exception:" + type(e).__name__
        STATS[key] += 1
        STATS["stock_calls"] += 1
        if STATS[key] <= 3:
            msg = f"[boltz_flash_triattn_patch] flash path raised {e!r} at {site} N={n}; falling back to stock for this call"
            print(msg, flush=True, file=sys.stderr)
            _STATE["errors"].append(msg[:500])
        return orig(self, x, mask=mask, chunk_size=chunk_size, use_kernels=use_kernels)
    STATS["flash_calls"] += 1
    STATS[f"flash|{site}|q={list(qshape)}|{qdt}|chunk={chunk_size}|use_kernels={bool(use_kernels)}"] += 1
    return out


def _patched_kernel_triangular_attn(q, k, v, tri_bias, mask, scale):
    """Path (a): replacement of primitives.kernel_triangular_attn (the cuEq call) -- same signature, per-call fallback to the original."""
    orig = _STATE["orig_kernel_fn"]
    STATS["kfn_calls"] += 1
    FT = _import_flash()
    ok, why = FT.flash_supported(q, k, v, tri_bias, mask)
    if ok and q.shape[-2] < _STATE["min_tokens"]:
        ok, why = False, f"n<{_STATE['min_tokens']}"
    if not ok:
        STATS[f"kfn_stock|{why}"] += 1
        return orig(q, k, v, tri_bias, mask, scale)
    try:
        out = FT.flash_triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale, exact_exp=_STATE["exact_exp"], use_libdevice=_STATE["use_libdevice"])
    except Exception as e:
        if is_oom(e): raise                              # a GPU out-of-memory error propagates: no fallback on out-of-memory
        STATS["kfn_stock|exception:" + type(e).__name__] += 1
        return orig(q, k, v, tri_bias, mask, scale)
    STATS[f"kfn_flash|q={list(q.shape)}"] += 1
    return out


def _patched_boltz2_forward(self, *a, **k):
    """Numerically inert wrapper: labels the TriangleAttention submodules once per model instance (for the per-call-site counters)."""
    if id(self) not in _LABELS:
        try:
            STATS["labelled_modules"] += label_modules(self)
        except Exception:
            pass
        _LABELS[id(self)] = "model"
    return _STATE["orig_boltz2_forward"](self, *a, **k)


def config() -> Dict[str, Any]:
    return {"version": PATCH_VERSION, "mode": _STATE["mode"], "applied": _STATE["applied"], "min_tokens": _STATE["min_tokens"], "exact_exp": _STATE["exact_exp"],
            "use_libdevice": _STATE["use_libdevice"], "flash_sha256": _STATE["flash_sha256"], "flash_file": _STATE["flash_file"],
            "forwarded_env": dict(_STATE["forwarded_env"]), "warmed": _STATE.get("warmed"),
            "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith("BOLTZ_TRIATTN") or k.startswith("PTX_TRIATTN")}}


def apply(force: bool = False, exact_exp: Optional[bool] = None, use_libdevice: Optional[bool] = None, min_tokens: Optional[int] = None,
          quiet: bool = False) -> bool:
    """Patch iff BOLTZ_TRIATTN=flash (or force=True). Returns True if the patch is active. Settings are frozen here (read once)."""
    mode = os.environ.get("BOLTZ_TRIATTN", "").strip().lower()
    if not force and mode != "flash":
        _STATE["mode"] = mode or "stock"
        return False
    with _LOCK:
        if _STATE["applied"]:
            return True
        fwd = _forward_env_knobs()
        _import_flash()   # import now (freezes flash_triattn's own env knobs)
        import boltz.model.layers.triangular_attention.attention as TA
        import boltz.model.layers.triangular_attention.primitives as TAP
        _STATE["exact_exp"] = (os.environ.get("BOLTZ_TRIATTN_EXACT", "1") != "0") if exact_exp is None else bool(exact_exp)
        _STATE["use_libdevice"] = (os.environ.get("BOLTZ_TRIATTN_LIBDEVICE", "0") == "1") if use_libdevice is None else bool(use_libdevice)
        _STATE["min_tokens"] = int(os.environ.get("BOLTZ_TRIATTN_MIN_TOKENS", str(DEFAULT_MIN_TOKENS))) if min_tokens is None else int(min_tokens)
        _STATE["forwarded_env"] = fwd
        _STATE["orig_forward"] = TA.TriangleAttention.forward
        TA.TriangleAttention.forward = _patched_forward                       # covers TriangleAttentionStartingNode (alias) and EndingNode (subclass)
        _STATE["orig_kernel_fn"] = TAP.kernel_triangular_attn
        TAP.kernel_triangular_attn = _patched_kernel_triangular_attn
        try:
            from boltz.model.models.boltz2 import Boltz2
            _STATE["orig_boltz2_forward"] = Boltz2.forward
            Boltz2.forward = _patched_boltz2_forward
        except Exception as e:  # labels are cosmetic
            _STATE["errors"].append(f"label hook not installed: {e!r}"[:300])
        _STATE["applied"] = True
        _STATE["mode"] = "flash"
        if os.environ.get("BOLTZ_TRIATTN_WARM", "") not in ("", "0"):
            warm_shapes()
    if os.environ.get("BOLTZ_TRIATTN_REPORT", "0") == "1":
        atexit.register(lambda: print("[boltz_flash_triattn_patch] " + json.dumps(report()), flush=True))
    if not quiet:
        print(f"[boltz_flash_triattn_patch] APPLIED flash_triattn version={PATCH_VERSION} min_tokens={_STATE['min_tokens']} exact_exp={_STATE['exact_exp']} "
              f"use_libdevice={_STATE['use_libdevice']} flash_sha256={(_STATE['flash_sha256'] or '?')[:12]}", flush=True, file=sys.stderr)
    return True


def remove() -> None:
    """Restore stock (for same-process A/B measurements)."""
    with _LOCK:
        if not _STATE["applied"]:
            return
        import boltz.model.layers.triangular_attention.attention as TA
        import boltz.model.layers.triangular_attention.primitives as TAP
        TA.TriangleAttention.forward = _STATE["orig_forward"]
        TAP.kernel_triangular_attn = _STATE["orig_kernel_fn"]
        if _STATE["orig_boltz2_forward"] is not None:
            try:
                from boltz.model.models.boltz2 import Boltz2
                Boltz2.forward = _STATE["orig_boltz2_forward"]
            except Exception:
                pass
        _STATE["applied"] = False
        _STATE["mode"] = "stock"



def warm_shapes(token_counts=None, c_z: int = 128, head_dim: int = 32, heads: int = 4, verbose: bool = True):
    """AOT-warm the Triton JIT so a batch of predictions never pays the one-off compile (~4-5 s per specialisation) mid-stream.
    Triton specialises the kernel on the constexpr config AND, for every integer argument (strides/sizes), on whether it is
    ==1 / divisible by 16 -- so the exact strides of the MODEL's q/k/v/bias views matter.  We therefore warm through the real
    module path: a boltz TriangleAttentionStartingNode + EndingNode (random weights, c_z=128, 4 heads x 32 as in Boltz-2) run on a
    synthetic pair tensor [1,N,N,c_z] under bf16 autocast with an all-ones mask, i.e. the same code path, dtypes and strides as
    inference (outputs discarded).  Also note Triton's on-disk cache (~/.triton/cache, or TRITON_CACHE_DIR) persists compiled
    variants across processes on the same machine: point TRITON_CACHE_DIR at a persistent directory to skip even this warm-up.
    Env: BOLTZ_TRIATTN_WARM="356,546,705,813" (an explicit list of token counts) or "1" (= ladder 320..1024 covering the usual
    divisibility classes).  Returns [(N, seconds)]."""
    import time as _t
    FT = _import_flash()
    if FT is None or not torch.cuda.is_available() or not _STATE["applied"]:
        return []
    if token_counts is None:
        spec = os.environ.get("BOLTZ_TRIATTN_WARM", "")
        if spec in ("", "0"):
            return []
        token_counts = [int(x) for x in spec.split(",") if x.strip().isdigit()] if spec != "1" else [320, 356, 400, 464, 512, 546, 600, 640, 705, 768, 813, 900, 1024]
    from boltz.model.layers.triangular_attention.attention import TriangleAttentionStartingNode, TriangleAttentionEndingNode
    mods = [TriangleAttentionStartingNode(c_z, head_dim, heads, inf=1e9).cuda().eval(), TriangleAttentionEndingNode(c_z, head_dim, heads, inf=1e9).cuda().eval()]
    done = []; mt = _STATE["min_tokens"]
    for n in token_counts:
        if n < mt:
            continue
        t0 = _t.time()
        try:
            z = torch.randn(1, n, n, c_z, device="cuda") * 0.1
            pm = torch.ones(1, n, n, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for m in mods:
                    for cs in (None, 128 if n > 256 else 512):      # boltz uses chunk_size_tri_attn 128 (>256 tok) / 512; the patch sees both call forms
                        m(z, mask=pm, chunk_size=cs, use_kernels=False)
            torch.cuda.synchronize()
            done.append((n, round(_t.time() - t0, 2)))
        except Exception as e:  # never fatal: the model call would JIT lazily instead
            _STATE["errors"].append(f"warm_shapes({n}): {e!r}"[:300]); done.append((n, None))
        finally:
            z = pm = None
    del mods; torch.cuda.empty_cache()
    _STATE["warmed"] = done
    if verbose:
        print(f"[boltz_flash_triattn_patch] warm_shapes: {done}", flush=True)
    return done

def report() -> Dict[str, Any]:
    try:
        FT = _import_flash()
        fls = FT.fast_launch_stats()
        fl = {k: v for k, v in fls.items() if k not in ("kernel_info", "last_kernel_info")}
        ki = fls.get("kernel_info", [])[:8]
    except Exception:
        fl, ki = None, None
    return {**config(), "stats": dict(sorted(STATS.items())), "n_flash": STATS.get("flash_calls", 0), "n_stock": STATS.get("stock_calls", 0),
            "fast_launch": fl, "kernel_info": ki, "errors": list(_STATE["errors"])[:20]}


# ------------------------------------------------------------------------------------------------------------------------------------------
def _preflight() -> int:
    """BOLTZ_TRIATTN=flash python boltz_flash_triattn_patch.py --preflight : random-weight TriangleAttention modules, flash vs stock under bf16 autocast."""
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("PREFLIGHT F2 FAIL no cuda"); return 1
    from boltz.model.layers.triangular_attention.attention import TriangleAttention
    if not apply():
        print("PREFLIGHT F2 NOT-APPLIED (set BOLTZ_TRIATTN=flash)"); return 1
    import triton
    dev = torch.device("cuda")
    worst = 0.0; lines = []
    for starting in (True, False):
        for n in (200, 400):          # 400 > chunk threshold 384 -> stock uses 128-row chunk_layer; 200 -> one 512 chunk
            m = TriangleAttention(128, 32, 4, starting=starting, inf=1e9).to(dev).eval()
            with torch.no_grad():
                for name, p in m.named_parameters():   # 'final'/'gating' inits are degenerate -> randomise so the check is not vacuous
                    p.normal_(0.0, 0.08)
                    if name.endswith("layer_norm.weight"):
                        p.add_(1.0)
            x = torch.randn(1, n, n, 128, device=dev) * 3.0
            mask = torch.ones(1, n, n, device=dev)
            cs = 128 if n > 384 else 512
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                ref = _STATE["orig_forward"](m, x, mask=mask, chunk_size=cs, use_kernels=False)
                out = m(x, mask=mask, chunk_size=cs, use_kernels=False)
                torch.cuda.synchronize()
            d = (out.float() - ref.float()).abs().max().item(); sref = ref.float().abs().max().item()
            worst = max(worst, d / max(sref, 1e-6))
            lines.append(f"start={starting} N={n} max|flash-stock|={d:.3e} max|stock|={sref:.3e}")
    st = report()
    okc = st["n_flash"] >= 4 and worst < 0.05
    print("PREFLIGHT F2 " + ("APPLIED" if okc else "FAIL") + f" flash_triattn version={PATCH_VERSION} min_tokens={_STATE['min_tokens']} exact_exp={_STATE['exact_exp']} "
          f"n_flash={st['n_flash']} n_stock={st['n_stock']} worst_rel={worst:.2e} torch={torch.__version__} triton={triton.__version__} "
          f"gpu={torch.cuda.get_device_name(0)} flash_sha256={(_STATE['flash_sha256'] or '?')[:12]} | " + " ; ".join(lines), flush=True)
    return 0 if okc else 1


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        sys.exit(_preflight())
    print(__doc__)
