"""boltz_waste_levers.py -- EXACT levers for the Boltz-2 2.2.1 MSA module's stock chunked inference paths (runtime monkeypatches,
inference only; the boltz wheel is untouched; default OFF = nothing patched = stock bytes).

Why these paths: at N > const.chunk_size_threshold (384 tokens) stock's MSAModule runs PairWeightedAveraging with the 8 heads sequential,
the MSA Transition with its hidden dim in 8 chunks of 32, the MSA-module pair Transition in 8 chunks of 64 and OuterProductMean in 8 chunks
of 4 (boltz/model/modules/trunkv2.py MSAModule.forward: chunk_heads_pwa / chunk_size_transition_msa / chunk_size_transition_z /
chunk_size_outer_product). No exact-tier lever serves these calls (the fused transition takes only unchunked calls in `exact`; the MSA
kernels are Tier 2), and inside the chunk loops stock re-does work whose result cannot change between iterations:

  chunkcast  Under bf16 autocast every `x @ w_slice.T` casts the fp32 LayerNorm output x to bf16 again — 16 casts of the [S,N,64] MSA tensor
             per Transition call and per PairWeightedAveraging call (2 per head x 8 heads), 16 of the [N,N,128] pair LayerNorm output per
             chunked pair Transition, 2 of the normed MSA tensor in OuterProductMean (proj_a, proj_b). The lever applies the cast autocast
             applies (`Tensor.to(torch.bfloat16)`, round-to-nearest-even — at::autocast::cached_cast is exactly `.to(dtype)` on activations)
             ONCE per call and hands the bf16 tensor to the same matmuls: identical operands (values, dtype, shape, contiguity), identical
             cuBLAS calls, identical bits. Active only when autocast is on with bf16 and the operand is fp32 (else the stock statement runs as is).
  opmmask    OuterProductMean's chunked path recomputes `num_mask` (the [B,N,N,1] pair count of unmasked MSA rows: 128 chunk products of
             [64,N,N] summed) on EVERY call — 16 calls per prediction (4 layers x 4 trunk passes) from the same `feats["msa_mask"]`; it
             depends only on that mask and the MSA tensor's dtype. The lever computes it with the identical statements once per (mask tensor,
             dtype) and reuses it; the cache holds a weak reference to the mask tensor object + its version counter, so a hit is only ever
             the same live, unmodified tensor (a new prediction's mask is a new object: no cross-item reuse is possible), 8 MB at 1,400 tokens.

Both are EXACT (bitwise = stock): the same arithmetic per output element, the redundant repeats removed (torch.equal on every output).

  opmdiv     OuterProductMean's chunk loop: `torch.einsum("bsic,bsjd->bijcd")` returns a PERMUTED view, so stock's `z.reshape(b,i,j,-1)` is a
             full copy of the bf16 [B,N,N,c*32] chunk, followed by `z / num_mask` into a new fp32 tensor. The lever divides the 5-D view by
             num_mask[..., None] straight into a contiguous fp32 buffer (`torch.div(..., out=)`: same element pairs, same bf16->fp32
             promotion, same correctly-rounded division) and the reshape becomes a view: one copy kernel less per chunk (8 per call, 16 calls).
             torch.equal on z and on the projected output.
  (chunkcast also casts the einsum's fp32 operands once per call: autocast re-casts the [B,S,N,c] slice AND the whole [B,S,N,32] `b` to
   bf16 on every one of the 8 chunks; each chunk now receives operands with the values/dtype/shape/contiguity autocast's cast produces.)

  ctorskip   FIRST-OF-PROCESS (model build): `Boltz2.load_from_checkpoint` constructs the module — 21 s of Boltz2.__init__ on the pinned stack, of
             which 18 s is boltz's `initialize.trunc_normal_init_` (scipy truncnorm ppf over ~250 M Transition weights, 472 calls) — and THEN
             load_state_dict(strict=True) overwrites every parameter. The draws matter only through the numpy RNG state they leave behind (stock
             seeds BEFORE construction; the worker keeps the post-construction state per seed). scipy's truncnorm takes rv_generic's generic
             inverse-cdf path: `U = random_state.uniform(size=size); Y = self._ppf(U)` — the ONLY draw is `uniform(size=size)` on numpy's
             global RandomState. The lever replaces trunc_normal_init_ by exactly that draw (no ppf, no copy into the soon-overwritten weight):
             post-construction RNG state (torch CPU / numpy / python / cuda) and every parameter and buffer identical to the stock build
             (unit test), the build several times faster per process (and per extra seed's constructor replay in the
             fast worker). Pinned by name: scipy major.minor, rv_generic._rvs source, truncnorm not overriding _rvs, trunc_normal_init_ source sha.

Switch (read ONCE at apply(); the kit's mode table assigns it): BOLTZ_WASTE=<comma list of {chunkcast,opmmask,opmdiv,ctorskip}>; unset/empty ->
nothing patched. Each lever reports state=on|off|skipped with a reason (report() / lever_lines()). A lever is `skipped` (stock statement
left in place, reason named) when another row word hands the same module to a kernel that replaces its forward wholesale:
BOLTZ_MSA_KERNELS containing `pwa`/`opm`/`transition` (boltz2_opt.msa_kernels, Tier 2 rows), or BOLTZ_XL set (the memory line's own
Transition handling) — and, independent of any word, when the forward found on the class at apply() is neither the stock function nor a
known chunk-delegating dispatcher (`skipped reason=foreign_forward:<module>`): the decision keys on the identity of the installed forward,
not on lever order. The Transition patch takes ONLY chunked calls and delegates every other call to the forward installed before it
(the fused-transition adapter's dispatcher when BOLTZ_TRANSITION is set; that adapter books chunked calls as `fallback_by=chunked`, i.e.
it never served them). Pin: boltz 2.2.1 — apply() refuses (raises) on another boltz version or when the three stock source files' sha256
differ from the pinned ones below (the bodies here are their verbatim copies plus the hoists).
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
import weakref
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SWITCH = "BOLTZ_WASTE"
LEVERS = ("chunkcast", "opmmask", "opmdiv", "ctorskip")
NAME = "W1.waste"
BOLTZ_VERSION = "2.2.1"
PINNED_SHA256 = {                                   # boltz 2.2.1 wheel: the stock files whose chunked bodies are copied below
    "boltz/model/layers/transition.py": "cc50a51a6aa669455420549de16e29558d7b96f8a7d8da6f69930ab9e738cf37",
    "boltz/model/layers/pair_averaging.py": "5d7662a723a5c59bf8411e1493670851ef0ec3f5e9af8e5e102aac5a8b8d015b",
    "boltz/model/layers/outer_product_mean.py": "c7966c776a9f5b8075f2d75c59ec88cc0ea2ba38694b568167c64ede619a774c",
}
_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {"applied": False, "levers": (), "state": {}, "orig": {}, "t_apply": None}
STATS: Dict[str, int] = {
    "transition_chunked_calls": 0, "transition_delegated_calls": 0, "transition_cast_hoisted": 0,
    "pwa_chunked_calls": 0, "pwa_delegated_calls": 0, "pwa_cast_hoisted": 0,
    "opm_chunked_calls": 0, "opm_delegated_calls": 0, "opm_cast_hoisted": 0, "opm_nummask_computed": 0, "opm_nummask_reused": 0,
    "opm_operand_cast_hoisted": 0, "opm_div_out": 0,
    "ctorskip_calls": 0, "ctorskip_elements": 0,
}


def _requested(environ=None) -> Tuple[str, ...]:
    env = os.environ if environ is None else environ
    words = [w.strip().lower() for w in env.get(SWITCH, "").split(",") if w.strip()]
    unknown = [w for w in words if w not in LEVERS]
    if unknown:
        raise RuntimeError(f"[boltz_waste_levers] {SWITCH}: unknown lever(s) {unknown}; known: {list(LEVERS)}")
    return tuple(w for w in LEVERS if w in words)


def _bf16_autocast() -> bool:
    """True inside a CUDA autocast region whose dtype is bf16 (the trunk's `torch.autocast("cuda", dtype=torch.bfloat16)`)."""
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16      # torch >= 2.4
    except TypeError:
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16              # older torch


def _q(x: torch.Tensor) -> torch.Tensor:
    """The cast autocast applies to an fp32 CUDA activation ahead of a bf16-listed op (at::autocast::cached_cast -> Tensor.to(dtype)), applied once."""
    return x.to(torch.bfloat16) if (x.dtype == torch.float32 and x.is_cuda) else x


def _check_pin() -> None:
    import importlib.metadata as im
    v = im.version("boltz")
    if v != BOLTZ_VERSION:
        raise RuntimeError(f"[boltz_waste_levers] boltz {v} installed; these levers carry boltz {BOLTZ_VERSION}'s bodies — refusing")
    import boltz
    root = os.path.dirname(os.path.dirname(boltz.__file__))
    for rel, want in PINNED_SHA256.items():
        with open(os.path.join(root, rel), "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != want:
            raise RuntimeError(f"[boltz_waste_levers] {rel} sha256 {got[:16]} != pinned {want[:16]} — the stock body changed; refusing")


# ----------------------------------------------------------------------------------------------------------------- Transition (chunked path)
def _make_transition_forward(prev):
    def forward(self, x, chunk_size: int = None):
        if chunk_size is None or self.training:                     # unchunked: whoever was installed before us (fused-transition dispatcher or stock)
            STATS["transition_delegated_calls"] += 1
            return prev(self, x, chunk_size) if chunk_size is not None else prev(self, x)
        STATS["transition_chunked_calls"] += 1
        # ---- stock boltz 2.2.1 Transition.forward, chunked branch, verbatim except `xq` (the autocast cast of x applied once) ----
        x = self.norm(x)
        if _bf16_autocast() and x.dtype == torch.float32:
            xq = _q(x); STATS["transition_cast_hoisted"] += 1
        else:
            xq = x
        for i in range(0, self.hidden, chunk_size):
            fc1_slice = self.fc1.weight[i : i + chunk_size, :]
            fc2_slice = self.fc2.weight[i : i + chunk_size, :]
            fc3_slice = self.fc3.weight[:, i : i + chunk_size]
            x_chunk = self.silu((xq @ fc1_slice.T)) * (xq @ fc2_slice.T)
            if i == 0:
                x_out = x_chunk @ fc3_slice.T
            else:
                x_out = x_out + x_chunk @ fc3_slice.T
        return x_out
    return forward


# ----------------------------------------------------------------------------------------------------------------- PairWeightedAveraging (heads sequential)
def _make_pwa_forward(prev):
    def forward(self, m, z, mask, chunk_heads: bool = False):
        if not (chunk_heads and not self.training):
            STATS["pwa_delegated_calls"] += 1
            return prev(self, m, z, mask, chunk_heads)
        STATS["pwa_chunked_calls"] += 1
        # ---- stock boltz 2.2.1 PairWeightedAveraging.forward, chunk_heads branch, verbatim except mq/zq (the autocast casts applied once) ----
        # Compute layer norms
        m = self.norm_m(m)
        z = self.norm_z(z)
        if _bf16_autocast() and (m.dtype == torch.float32 or z.dtype == torch.float32):
            mq, zq = _q(m), _q(z); STATS["pwa_cast_hoisted"] += 1
        else:
            mq, zq = m, z
        # Compute heads sequentially
        sliced_weight_proj_m = self.proj_m.weight.view(self.num_heads, self.c_h, self.c_m)
        sliced_weight_proj_g = self.proj_g.weight.view(self.num_heads, self.c_h, self.c_m)
        sliced_weight_proj_z = self.proj_z.weight.view(self.num_heads, 1, self.c_z)
        sliced_weight_proj_o = self.proj_o.weight.view(self.c_m, self.num_heads, self.c_h).permute(1, 0, 2)
        for head_idx in range(self.num_heads):
            sliced_weight_proj_m_head = sliced_weight_proj_m[head_idx]
            sliced_weight_proj_g_head = sliced_weight_proj_g[head_idx]
            sliced_weight_proj_z_head = sliced_weight_proj_z[head_idx]
            sliced_weight_proj_o_head = sliced_weight_proj_o[head_idx]

            # Project input tensors
            v: torch.Tensor = mq @ sliced_weight_proj_m_head.T
            v = v.reshape(*v.shape[:3], 1, self.c_h)
            v = v.permute(0, 3, 1, 2, 4)

            # Compute weights
            b: torch.Tensor = zq @ sliced_weight_proj_z_head.T
            b = b.permute(0, 3, 1, 2)
            b = b + (1 - mask[:, None]) * -self.inf
            w = torch.softmax(b, dim=-1)

            # Compute gating
            g: torch.Tensor = mq @ sliced_weight_proj_g_head.T
            g = g.sigmoid()

            # Compute output
            o = torch.einsum("bhij,bhsjd->bhsid", w, v)
            o = o.permute(0, 2, 3, 1, 4)
            o = o.reshape(*o.shape[:3], 1 * self.c_h)
            o_chunks = g * o
            if head_idx == 0:
                o_out = o_chunks @ sliced_weight_proj_o_head.T
            else:
                o_out += o_chunks @ sliced_weight_proj_o_head.T
        return o_out
    return forward


# ----------------------------------------------------------------------------------------------------------------- OuterProductMean (chunked path)
_NUMMASK: Dict[int, Tuple[Any, int, torch.dtype, torch.Tensor]] = {}     # id(mask tensor) -> (weakref, version, m dtype, num_mask); at most a few entries


def _ver(t: torch.Tensor) -> int:
    """The tensor's version counter (bumped by in-place writes); inference-mode tensors carry none (-1: identity + liveness is the key then —
    boltz never writes feats["msa_mask"] in place)."""
    try:
        return t._version
    except RuntimeError:
        return -1


def _nummask_lookup(mask_in: torch.Tensor, dtype: torch.dtype):
    for k in [k for k, (r, _v, _d, _t) in _NUMMASK.items() if r() is None]:
        del _NUMMASK[k]                                          # the tensor died (its prediction finished): drop the entry
    e = _NUMMASK.get((id(mask_in), dtype))
    if e is not None and e[0]() is mask_in and e[1] == _ver(mask_in) and e[2] == dtype:
        return e[3]
    return None


def _nummask_store(mask_in: torch.Tensor, dtype: torch.dtype, num_mask: torch.Tensor) -> None:
    if len(_NUMMASK) > 8:
        _NUMMASK.clear()
    key = (id(mask_in), dtype)
    # the weakref's callback drops the entry (and with it the cached [B,N,N,1] num_mask) the moment the mask tensor object is deallocated —
    # i.e. when the prediction's feats dict dies; nothing here holds the [B,S,N] mask itself (weak reference only)
    _NUMMASK[key] = (weakref.ref(mask_in, lambda _r, _k=key: _NUMMASK.pop(_k, None)), _ver(mask_in), dtype, num_mask)


def _make_opm_forward(prev, cast_hoist: bool, mask_cache: bool, div_out: bool = False):
    def forward(self, m, mask, chunk_size: int = None):
        if not (chunk_size is not None and not self.training):
            STATS["opm_delegated_calls"] += 1
            return prev(self, m, mask, chunk_size) if chunk_size is not None else prev(self, m, mask)
        STATS["opm_chunked_calls"] += 1
        mask_in = mask
        # ---- stock boltz 2.2.1 OuterProductMean.forward, chunked branch, verbatim except mq (cast once) and the num_mask reuse ----
        # Expand mask
        mask = mask.unsqueeze(-1).to(m)

        # Compute projections
        m = self.norm(m)
        if cast_hoist and _bf16_autocast() and m.dtype == torch.float32:
            mq = _q(m); STATS["opm_cast_hoisted"] += 1
        else:
            mq = m
        a = self.proj_a(mq) * mask
        b = self.proj_b(mq) * mask

        # Compute pairwise mask
        num_mask = _nummask_lookup(mask_in, mask.dtype) if mask_cache else None
        if num_mask is None:
            for i in range(0, mask.shape[1], 64):
                if i == 0:
                    num_mask = (
                        mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
                    ).sum(1)
                else:
                    num_mask += (
                        mask[:, i : i + 64, None, :] * mask[:, i : i + 64, :, None]
                    ).sum(1)
            num_mask = num_mask.clamp(min=1)
            if mask_cache:
                _nummask_store(mask_in, mask.dtype, num_mask); STATS["opm_nummask_computed"] += 1
        else:
            STATS["opm_nummask_reused"] += 1

        # chunkcast (v2): autocast casts the einsum's fp32 operands to bf16 on EVERY chunk — the [B,S,N,c] slice `a_chunk` and the whole
        # [B,S,N,32] `b` (8x per call); the cast is a pure function of the tensor, so cast both ONCE and hand each chunk operands with the
        # values, dtype, shape and (contiguous) layout autocast's per-chunk cast would have produced.
        if cast_hoist and _bf16_autocast() and a.dtype == torch.float32 and b.dtype == torch.float32:
            a16 = _q(a); b16 = _q(b); STATS["opm_operand_cast_hoisted"] += 1
        else:
            a16, b16 = a, b
        # Compute squentially in chunks
        for i in range(0, self.c_hidden, chunk_size):
            a_chunk = a16[:, :, :, i : i + chunk_size]
            if a16 is not a:
                a_chunk = a_chunk.contiguous()          # autocast's cast of the fp32 slice is a contiguous bf16 tensor: same layout here
            sliced_weight_proj_o = self.proj_o.weight[
                :, i * self.c_hidden : (i + chunk_size) * self.c_hidden
            ]

            z = torch.einsum("bsic,bsjd->bijcd", a_chunk, b16)
            if div_out:
                # opmdiv: the einsum returns a permuted view; stock's reshape copies it (bf16 [B,N,N,c*32]) and then divides into a new fp32
                # tensor. Divide the 5-D view by num_mask straight into a contiguous fp32 buffer instead (same element pairs, same promotion,
                # same division), and the reshape becomes a view.
                zq = torch.empty(z.shape, dtype=torch.result_type(z, num_mask), device=z.device)
                torch.div(z, num_mask.unsqueeze(-1), out=zq)
                z = zq.reshape(*zq.shape[:3], -1); STATS["opm_div_out"] += 1
            else:
                z = z.reshape(*z.shape[:3], -1)
                z = z / num_mask

            # Project to output
            if i == 0:
                z_out = z.to(m) @ sliced_weight_proj_o.T
            else:
                z_out = z_out + z.to(m) @ sliced_weight_proj_o.T

        z_out = z_out + self.proj_o.bias  # add bias
        return z_out
    return forward


# ----------------------------------------------------------------------------------------------------------------- ctorskip (model-build init draws)
CTORSKIP_PINS = {
    "scipy_series": ("1.13",),                                                    # rv_generic._rvs = uniform(size) + _ppf at these versions (source also checked)
    "rvs_stmt": "U = random_state.uniform(size=size)",                            # the one RNG draw in scipy.stats._distn_infrastructure.rv_generic._rvs
    "trunc_normal_init_sha256": None,                                             # filled lazily from the vendored 2.2.1 source text below (identity check by text)
}
_TRUNC_NORMAL_INIT_SRC_221 = (
    'def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):\n'
    '    shape = weights.shape\n'
    '    f = _calculate_fan(shape, fan)\n'
    '    scale = scale / max(1, f)\n'
    '    a = -2\n'
    '    b = 2\n'
    '    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)\n'
    '    size = _prod(shape)\n'
    '    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)\n'
    '    samples = np.reshape(samples, shape)\n'
    '    with torch.no_grad():\n'
    '        weights.copy_(torch.tensor(samples, device=weights.device))\n')


def _ctorskip_check() -> str:
    """'' when the pinned facts hold in this process, else the refusal reason (the lever then stays off BY NAME; stock init runs)."""
    import inspect
    try:
        import scipy
        from scipy.stats import truncnorm
        from scipy.stats._distn_infrastructure import rv_generic
        import boltz.model.layers.initialize as INIT
    except Exception as e:  # noqa: BLE001
        return f"import:{type(e).__name__}"
    if not any(scipy.__version__.startswith(v) for v in CTORSKIP_PINS["scipy_series"]):
        return f"scipy_version:{scipy.__version__}"
    try:
        if CTORSKIP_PINS["rvs_stmt"] not in inspect.getsource(rv_generic._rvs):
            return "rv_generic._rvs:source_changed"
        if type(truncnorm)._rvs is not rv_generic._rvs:                        # truncnorm_gen must NOT override the generic inverse-cdf sampler
            return "truncnorm._rvs:overridden"
        src = inspect.getsource(INIT.trunc_normal_init_)
    except Exception as e:  # noqa: BLE001
        return f"inspect:{type(e).__name__}"
    if src.strip() != _TRUNC_NORMAL_INIT_SRC_221.strip():
        return "trunc_normal_init_:source_changed"
    return ""


def _trunc_normal_init_rng_only(weights, scale=1.0, fan="fan_in"):
    """ctorskip stand-in for boltz.model.layers.initialize.trunc_normal_init_: performs the numpy RNG draw scipy's truncnorm.rvs performs
    (rv_generic._rvs: random_state.uniform(size=size) on the global RandomState) and nothing else — no ppf, no copy into `weights`, whose
    values load_state_dict(strict=True) replaces. `truncnorm.std(...)` in the stock body draws nothing."""
    import boltz.model.layers.initialize as INIT
    shape = weights.shape
    INIT._calculate_fan(shape, fan)                                              # the stock ValueError for an invalid `fan`, before any draw (as stock)
    size = INIT._prod(shape)
    np.random.mtrand._rand.uniform(size=size)
    STATS["ctorskip_calls"] += 1; STATS["ctorskip_elements"] += int(size)


# ----------------------------------------------------------------------------------------------------------------- apply / remove / report
STOCK_MODULE = {"transition": "boltz.model.layers.transition", "pwa": "boltz.model.layers.pair_averaging", "opm": "boltz.model.layers.outer_product_mean"}
# forwards installed by another lever that are KNOWN to hand the chunked branch to the stock body (so wrapping them and taking the chunked
# calls first changes nothing but who books them): the fused-transition adapter's dispatcher in its `exact` variant (chunked -> fallback_by=chunked).
DELEGATING_MODULE = {"transition": {"boltz2_opt.transition"}, "pwa": set(), "opm": set()}


def _forward_origin(fn) -> str:
    """The defining module of the function object currently installed as the class forward ('?' if it cannot be told)."""
    return getattr(fn, "__module__", None) or "?"


def _transition_adapter_variant() -> str:
    """The fused-transition adapter's active variant ('exact' | 'fast' | '' when absent/unapplied) — read from the adapter itself, env as fallback."""
    try:
        import boltz2_opt.transition as T5
        v = (getattr(T5, "_STATE", {}) or {}).get("variant") or ""
        if v:
            return str(v).lower()
    except Exception:
        pass
    return os.environ.get("BOLTZ_TRANSITION", "").strip().lower()


def _skips(current: Dict[str, Any], environ=None) -> Dict[str, str]:
    """module -> reason its forward is left alone. Decided from the IDENTITY of the forward object currently installed on the class (stock, a known
    chunk-delegating dispatcher, or foreign), refined by the row words that hand a module to a wholesale replacement."""
    env = os.environ if environ is None else environ
    out = {}
    for unit, fn in current.items():
        origin = _forward_origin(fn)
        if origin == STOCK_MODULE[unit]:
            continue
        if origin in DELEGATING_MODULE[unit]:
            if unit == "transition" and _transition_adapter_variant() == "fast":
                out[unit] = "transition_fast_serves_chunked (boltz2_opt.transition variant=fast)"
            continue
        out[unit] = f"foreign_forward:{origin}"
    mk = {w.strip().lower() for w in env.get("BOLTZ_MSA_KERNELS", "").split(",") if w.strip()}
    if env.get("BOLTZ_XL", "").strip():
        for u in ("transition", "pwa", "opm"):
            out.setdefault(u, "BOLTZ_XL set (memory line owns these modules)")
    if "transition" in mk:
        out.setdefault("transition", "BOLTZ_MSA_KERNELS=transition serves the MSA transition")
    if "pwa" in mk:
        out.setdefault("pwa", "BOLTZ_MSA_KERNELS=pwa replaces PairWeightedAveraging")
    if "opm" in mk:
        out.setdefault("opm", "BOLTZ_MSA_KERNELS=opm replaces OuterProductMean")
    return out


def apply(environ=None) -> Tuple[str, ...]:
    """Patch per the row word; returns the levers switched on. Idempotent. Raises on an unknown lever name or a pin mismatch (never a silent fallback)."""
    with _LOCK:
        if _STATE["applied"]:
            return _STATE["levers"]
        levers = _requested(environ)
        state = {lv: {"state": "off", "reason": "not in " + SWITCH} for lv in LEVERS}
        if not levers:
            _STATE.update(applied=True, levers=(), state=state, t_apply=time.time())
            return ()
        _check_pin()
        from boltz.model.layers import transition as TRN
        from boltz.model.layers import pair_averaging as PWA
        from boltz.model.layers import outer_product_mean as OPM
        current = {"transition": TRN.Transition.forward, "pwa": PWA.PairWeightedAveraging.forward, "opm": OPM.OuterProductMean.forward}
        skips = _skips(current, environ)
        _STATE["over"] = {u: _forward_origin(f) for u, f in current.items()}
        O = _STATE["orig"]
        patched: List[str] = []
        cast = "chunkcast" in levers
        if cast:
            done, skipped = [], []
            if "transition" in skips:
                skipped.append("transition:" + skips["transition"])
            else:
                O["transition_fwd"] = TRN.Transition.forward; TRN.Transition.forward = _make_transition_forward(O["transition_fwd"]); done.append("Transition")
            if "pwa" in skips:
                skipped.append("pwa:" + skips["pwa"])
            else:
                O["pwa_fwd"] = PWA.PairWeightedAveraging.forward; PWA.PairWeightedAveraging.forward = _make_pwa_forward(O["pwa_fwd"]); done.append("PairWeightedAveraging")
            state["chunkcast"] = {"state": "on" if done else "skipped", "patched": done, "skipped": skipped,
                                  "reason": "; ".join(skipped) if (skipped and not done) else ""}
            patched += done
        opm_cast, opm_mask = cast and "opm" not in skips, ("opmmask" in levers) and "opm" not in skips
        opm_div = ("opmdiv" in levers) and "opm" not in skips
        if opm_cast or opm_mask or opm_div:
            O["opm_fwd"] = OPM.OuterProductMean.forward
            OPM.OuterProductMean.forward = _make_opm_forward(O["opm_fwd"], cast_hoist=opm_cast, mask_cache=opm_mask, div_out=opm_div); patched.append("OuterProductMean")
            if cast:
                state["chunkcast"]["patched"] = state["chunkcast"].get("patched", []) + ["OuterProductMean(proj casts)"]
        if "opmmask" in levers:
            state["opmmask"] = {"state": "on", "reason": ""} if opm_mask else {"state": "skipped", "reason": skips.get("opm", "")}
        if "opmdiv" in levers:
            state["opmdiv"] = {"state": "on", "reason": ""} if opm_div else {"state": "skipped", "reason": skips.get("opm", "")}
        elif cast and "opm" in skips:
            state["chunkcast"].setdefault("skipped", []).append("opm:" + skips["opm"])
        if "ctorskip" in levers:
            why = _ctorskip_check()
            if why:
                state["ctorskip"] = {"state": "skipped", "reason": why}
            else:
                import boltz.model.layers.initialize as INIT
                O["trunc_normal_init_"] = INIT.trunc_normal_init_; INIT.trunc_normal_init_ = _trunc_normal_init_rng_only
                state["ctorskip"] = {"state": "on", "reason": ""}; patched.append("initialize.trunc_normal_init_")
        _STATE.update(applied=True, levers=levers, state=state, t_apply=time.time(), patched=patched)
    if (os.environ if environ is None else environ).get("BOLTZ_WASTE_VERBOSE", "1") == "1":
        for ln in lever_lines():
            print(ln, flush=True)
    return levers


def remove() -> None:
    with _LOCK:
        if not _STATE["applied"]:
            return
        from boltz.model.layers import transition as TRN
        from boltz.model.layers import pair_averaging as PWA
        from boltz.model.layers import outer_product_mean as OPM
        O = _STATE["orig"]
        if "transition_fwd" in O:
            TRN.Transition.forward = O["transition_fwd"]
        if "pwa_fwd" in O:
            PWA.PairWeightedAveraging.forward = O["pwa_fwd"]
        if "trunc_normal_init_" in O:
            import boltz.model.layers.initialize as INIT
            INIT.trunc_normal_init_ = O["trunc_normal_init_"]
        if "opm_fwd" in O:
            OPM.OuterProductMean.forward = O["opm_fwd"]
        _NUMMASK.clear()
        _STATE.update(applied=False, levers=(), state={}, orig={})


def report() -> Dict[str, Any]:
    return {"name": NAME, "switch": SWITCH, "applied": _STATE["applied"], "levers": list(_STATE["levers"]), "state": dict(_STATE["state"]),
            "patched": list(_STATE.get("patched", [])), "over": dict(_STATE.get("over") or {}), "nummask_entries": len(_NUMMASK),
            "stats": dict(STATS), "torch": torch.__version__, "boltz": BOLTZ_VERSION}


def lever_lines() -> List[str]:
    """One `[boltz2-opt] LEVER name=W1.<lever> state=on|off|skipped …` line per lever (an ablated/absent lever is named, never dropped silently)."""
    out = []
    for lv in LEVERS:
        st = _STATE["state"].get(lv, {"state": "off", "reason": "not applied"})
        extra = ""
        if st.get("patched"):
            extra += " patched=" + ",".join(st["patched"])
        if st.get("skipped"):
            extra += " skipped=" + "|".join(s.replace(" ", "_") for s in st["skipped"])
        over = _STATE.get("over") or {}
        if st.get("state") == "on" and lv == "ctorskip":
            extra += f" calls={STATS['ctorskip_calls']} elements={STATS['ctorskip_elements']}"
        elif st.get("state") == "on" and over:
            units = ("transition", "pwa", "opm") if lv == "chunkcast" else ("opm",)   # opmmask / opmdiv act on OPM only
            extra += " over=" + ",".join(f"{u}:{over.get(u, '?')}" for u in units)
        reason = st.get("reason") or ""
        out.append(f"[boltz2-opt] LEVER name=W1.{lv} state={st['state']} class=exact switch={SWITCH}" + (f" reason={reason.replace(' ', '_')}" if reason else "") + extra)
    return out


def verdict() -> Tuple[bool, str]:
    """Fail-closed gate for the run's evidence check: a lever switched on must have acted (its counters moved) — else the run is not the mode it names."""
    if not _STATE["applied"]:
        return True, "not-applied"
    st = _STATE["state"]
    problems = []
    if st.get("chunkcast", {}).get("state") == "on" and (STATS["transition_cast_hoisted"] + STATS["pwa_cast_hoisted"] + STATS["opm_cast_hoisted"]) == 0 and (STATS["transition_chunked_calls"] + STATS["pwa_chunked_calls"]) > 0:
        problems.append("chunkcast:on-but-never-hoisted")
    if st.get("opmmask", {}).get("state") == "on" and STATS["opm_chunked_calls"] > 1 and STATS["opm_nummask_reused"] == 0:
        problems.append("opmmask:on-but-never-reused")
    if st.get("opmdiv", {}).get("state") == "on" and STATS["opm_chunked_calls"] > 0 and STATS["opm_div_out"] == 0:
        problems.append("opmdiv:on-but-never-used")
    if st.get("ctorskip", {}).get("state") == "on" and STATS["ctorskip_calls"] == 0:
        problems.append("ctorskip:on-but-no-constructor-ran-after-apply")
    return (not problems), (",".join(problems) or "ok")
