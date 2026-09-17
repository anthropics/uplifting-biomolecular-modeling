"""infopt_graphs.protenix.graphed — CUDA-graph capture of Protenix inference (lever `sg`: the diffusion denoiser step; the trunk capture is a
library option no kit mode turns on).

What is captured
  (a) DIFFUSION DENOISER STEP  (protenix/model/generator.py::sample_diffusion, 200 steps, Algorithm 18):
      one graph per static signature (batch_shape, chunk_n_sample, N_atom, N_token, dtype, flags).  Inside the graph:
      centre + random roto-translation (Alg. 19, deterministic part), noise injection x_noisy = x + lambda*dt*eps, the
      denoiser network call, and the Euler update, writing the new x_l IN PLACE into the static buffer.  Every random
      number is drawn OUTSIDE the graph, per step, in exactly the stock order and with the stock calls
      (scipy Rotation.random on the numpy global RNG -> torch.randn for the translation -> torch.randn for eps), so the
      random stream consumed is bit-identical to stock `sample_diffusion`.  Step scalars (t_hat, c_tau, delta_noise_level,
      dt) are computed outside exactly as stock (same torch ops on the same schedule tensor) and copied into 0-d static
      buffers.  The gamma decision `c_tau > gamma_min` stays on the host (stock: a Python float comparison) — it is a
      per-step Python branch, so the schedule's gamma pattern is part of the step's static signature.
  (b) TRUNK RECYCLE BODY: PairformerStack.forward (48 blocks) per recycle, one graph per (N_token, dtype, kernel flags).
      The MSA module (random subsample depth -> dynamic shapes), the recycling embedding with the MC-dropout F.dropout mask,
      and the input embedder stay EAGER, so the MSA randperm/randint and the dropout mask are drawn at exactly the stock
      points of the RNG stream.  The graphed stack consumes no RNG (asserted by RNGGuard at warm-up).

Fidelity: MSA unchanged (Protenix per-recycle subsampling untouched, drawn eagerly); step / cycle / sample counts as configured;
seeds: stock seed_everything + stock draw order; precision: stock (bf16 autocast trunk with
the autocast weight-cast cache disabled inside the captured region = identical casts recorded as kernels; fp32+TF32
denoiser); kernels: the SAME kernels as eager (cuEquivariance trimul/triattn, fast_layernorm, cuDNN/FMHA, Triton) —
a CUDA graph records the eager kernel sequence and replays it; checkpoint bytes unchanged.  Exact-shape mode
(default) pads nothing.  Expected numerics: identical up to GPU nondeterminism (kernel selection can differ between an
eager call and a capture only if cuBLAS/cuDNN heuristics depend on the stream/workspace).

Lever sampler_prep (install(..., sampler_prep=True); module sampler_prep.py): the loop's per-item / per-step HOST path — pinned non-blocking
rotation upload, a shape-keyed graph cache with an on-device comparison of the integer/mask feature values, one warm-up pass, the poison
self-test once per process, the previous entry's static buffers released before a new capture and its memory pool re-used when the new
signature fits (else returned to the device first), vectorised step scalars.  No kernel, RNG draw or graph changes: exact by construction.

Disable: install(..., sampler=False, trunk=False), or never call install().
"""
from __future__ import annotations
from opt_core.oom import is_oom
from . import sampler_prep as _sp          # lever sampler_prep: host path of the loop; see sampler_prep.py

import hashlib
import logging
import os
import time
import types
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from opt_core.tools.graph_audit.audit import RNGGuard, SyncCensus
from ..core import GraphedFunction, POOLS, _autocast_nocache, capture_context, copy_into, static_like

log = logging.getLogger("infopt_graphs.protenix")

# ---------------------------------------------------------------------------------------------------------------------
# (a) graphed diffusion sampler
# ---------------------------------------------------------------------------------------------------------------------


class EmptyCacheGuard:
    """torch.cuda.empty_cache() guard.  Protenix calls empty_cache inside a prediction; the call releases cached caching-allocator
    segments to the driver, and with >= 2 live private-pool sampler graphs in one process this faults on the default allocator
    ('illegal memory access'): memory referenced from an older capture's eager warm-up lives in the regular pool and is unmapped by
    the release.  Policy: while >= 2 sampler graphs are cached the release is DEFERRED (counted) and executed once the cache is back to
    <= 1 live graph (or at uninstall/exit).  Deferral keeps the cached segments; CEILING: if torch.cuda.memory_reserved() exceeds the
    ceiling (0.60 x total device memory) while a release is requested, the sampler cache is first evicted down to ONE live graph and
    THEN the release runs (never the reverse order).  With max_entries=1 (the loop's setting whenever the hoist is bound) the guard
    never defers (live graphs <= 1)."""
    def __init__(self, loop):
        self.loop = loop; self.orig = torch.cuda.empty_cache; self.installed = False
        self.deferred_now = 0; self.deferred_total = 0; self.flushed = 0; self.passed = 0; self.ceiling_evictions = 0
        self.peak_reserved_deferring = 0; self.reserved_at_first_defer = None
        self.enabled = True
        try:
            tot = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory if torch.cuda.is_available() else 0
        except Exception:
            tot = 0
        self.ceiling_bytes = int(0.60 * tot)                     # defer empty_cache while live graphs hold the pool, up to 60 % of the card reserved
        import atexit; atexit.register(self._atexit)
    def _live(self):
        return sum(1 for e in self.loop.entries.values() if e.get("graph") is not None)
    def __call__(self):
        if not (self.enabled and self._live() >= 2):
            self.passed += 1; return self.orig()
        reserved = torch.cuda.memory_reserved()
        if self.ceiling_bytes and reserved > self.ceiling_bytes:
            # over the ceiling: evict sampler graphs down to ONE live graph, THEN release (never the reverse)
            torch.cuda.synchronize()
            while self._live() > 1 and len(self.loop.order) > 1:
                self.loop._evict_oldest(); self.ceiling_evictions += 1
            self.loop.stats["events"].append({"event": "empty_cache_guard_ceiling", "reserved_gb": round(reserved / 2**30, 2), "ceiling_gb": round(self.ceiling_bytes / 2**30, 2)})
            self.deferred_now = 0; self.flushed += 1; return self.orig()
        if self.reserved_at_first_defer is None: self.reserved_at_first_defer = reserved
        self.peak_reserved_deferring = max(self.peak_reserved_deferring, reserved)
        self.deferred_now += 1; self.deferred_total += 1; return None
    def install(self):
        if self.installed or not self.enabled: return self
        torch.cuda.empty_cache = self
        try:
            import torch.cuda.memory as _m; _m.empty_cache = self
        except Exception: pass
        self.installed = True; return self
    def flush_if_safe(self):
        if self.deferred_now and self._live() <= 1:
            self.orig(); self.flushed += 1; self.deferred_now = 0
    def uninstall(self):
        if self.installed:
            torch.cuda.empty_cache = self.orig
            try:
                import torch.cuda.memory as _m; _m.empty_cache = self.orig
            except Exception: pass
            self.installed = False
        if self.deferred_now:
            try: self.orig()
            except Exception: pass
            self.flushed += 1; self.deferred_now = 0
    def report(self):
        return {"enabled": self.enabled, "installed": self.installed, "empty_cache_deferred": self.deferred_total, "deferred_pending": self.deferred_now, "flushed": self.flushed,
                "passed_through": self.passed, "ceiling_gb": round(self.ceiling_bytes / 2**30, 2), "ceiling_evictions": self.ceiling_evictions,
                "reserved_gb_at_first_defer": (round(self.reserved_at_first_defer / 2**30, 2) if self.reserved_at_first_defer is not None else None),
                "peak_reserved_gb_while_deferring": round(self.peak_reserved_deferring / 2**30, 2)}
    def _atexit(self):
        try:
            r = self.report()
            print(f"[infopt_graphs] SUMMARY sampler_cache max_entries={self.loop.max_entries} live_graphs={self._live()} captures={self.loop.stats.get('captures')} "
                  f"evictions={self.loop.stats.get('evictions')} empty_cache_deferred={r['empty_cache_deferred']} empty_cache_flushed={r['flushed']} empty_cache_passed={r['passed_through']} "
                  f"ceiling_gb={r['ceiling_gb']} ceiling_evictions={r['ceiling_evictions']} reserved_gb_at_first_defer={r['reserved_gb_at_first_defer']} peak_reserved_gb_while_deferring={r['peak_reserved_gb_while_deferring']}", flush=True)
        except Exception:
            pass


def _static_bytes(ent) -> int:
    """Bytes held by an entry's static tensors (inputs copies `st`, conditioning `cond`, bias-cache buffers `bc`)."""
    seen = set(); total = 0
    def walk(o):
        nonlocal total
        if isinstance(o, torch.Tensor):
            if o.is_cuda and o.data_ptr() not in seen:
                seen.add(o.data_ptr()); total += o.numel() * o.element_size()
        elif isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o: walk(v)
    for k in ("st", "cond", "bc"):
        walk(ent.get(k))
    return int(total)


def _tensor_value_digest(v: torch.Tensor, digest_cache: Optional[dict] = None) -> str:
    """A content digest of v's values (one device->host sync + one sha256). Memoized by tensor handle (data_ptr,
    shape, dtype, the in-place-write counter `_version`) in the caller's digest_cache, if given -- one physical tensor
    buffer is digested once no matter how many times `_chunk()` reads it within the SAME sample() call (diffusion_chunk_size
    can split one item's N_sample across several `_chunk()` calls, all sharing the same input_feature_dict tensors)."""
    if digest_cache is None:
        return hashlib.sha256(v.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()
    ident = (v.data_ptr(), tuple(v.shape), str(v.dtype), v._version)
    d = digest_cache.get(ident)
    if d is None:
        d = hashlib.sha256(v.detach().to("cpu").contiguous().numpy().tobytes()).hexdigest()
        digest_cache[ident] = d
    return d


def _input_feature_key(input_feature_dict, digest_cache: Optional[dict] = None) -> tuple:
    """The input_feature_dict-derived part of a graph's cache key: (name, shape, dtype) for every tensor, PLUS a content
    digest for every NON-FLOATING one (an integer index or a bool/int mask a captured graph bakes into its kernel-launch
    parameters via gather / scatter / index_select -- e.g. atom_to_token_idx). Two items with equal shapes but different
    index VALUES must never collide on the same key and silently replay the wrong graph. Floating tensors (coordinates,
    continuous features) key by shape/dtype alone: the captured graph is a normal differentiable computation on their
    values, valid to replay for any values of the same shape. `digest_cache`: see _tensor_value_digest -- pass a dict
    scoped to one sample() call so a multi-chunk item digests each tensor once, not once per chunk."""
    return tuple(sorted((k, tuple(v.shape), str(v.dtype),
                         _tensor_value_digest(v, digest_cache) if not v.is_floating_point() and not v.is_complex() else None)
                        for k, v in input_feature_dict.items() if isinstance(v, torch.Tensor)))


class GraphedDenoiseLoop:
    """Owns the static buffers and the captured step graph per signature.  One instance per Protenix model."""

    def __init__(self, family: str = "protenix_sampler", pool: str = "shared", rng_guard: bool = True, sync_audit: bool = True,
                 max_entries: int = 8, max_pool_bytes: int = 0, max_tokens: int = 0, prep=None):
        self.family = family
        self.pool_mode = pool
        # --- MEMORY BUDGET (a long run of large-token items must not exhaust the pool): entries are evicted by COUNT
        # (max_entries) AND by BYTES (max_pool_bytes, 0 = unlimited): bytes = pool growth during capture + the entry's static buffers.
        # max_tokens (0 = off): predictions with N_token > max_tokens BYPASS the graphed loop and run the STOCK sample_diffusion
        # (the function this loop replaces; bias cache off for that call).
        self.max_pool_bytes = int(max_pool_bytes or 0)
        self.max_tokens = int(max_tokens or 0)
        self.evict_empty_cache = True
        self.rng_guard = rng_guard
        self.sync_audit = sync_audit
        self.max_entries = max_entries
        self.teacher_forced_steps: Tuple[int, ...] = ()   # step indices at which to run the teacher-forced check (costly: +2 eager steps each)
        self.disable_capture = False
        self.nan_check_steps: Tuple[int, ...] = ()        # diagnostic: assert finite x_l after these steps (one sync each)
        self.teacher_forced: list = []                      # [{step, N_atom, d_graph_vs_eager, d_eager_vs_eager, ...}]
        self.entries: Dict[Tuple, Dict[str, Any]] = {}
        self.order = []
        self.stats: Dict[str, Any] = {"captures": 0, "replays": 0, "warmup_steps": 0, "eager_steps": 0, "events": [], "record_steps": 0, "bc_audit_steps": 0, "poison": [],
                                      "bypass": 0, "evictions": 0, "evicted_bytes": 0}
        # --- static bias cache: the DiT hoist bound as `biascache` (dit_hoist.Hoist) — None = off (stock per-step recomputation)
        self.biascache = None
        self.biascache_audit_steps: Tuple[int, ...] = ()   # eager audit steps (stock ops vs the static buffers, torch.equal)
        self.poison_check = True                            # once per capture: prove the graph reads the static buffers
        self.prep = prep if prep is not None else _sp.PrepConfig(False)   # lever sampler_prep: which host-path parts are on (all six or none; install(sampler_prep=...))
        self._prep_chain = None                          # [pool_chain] the previous entry's graph, alive until the next capture has ended
        self._prep_renewed = False                       # [pool_chain] a renewed capture (cache emptied before it) still gets the post-capture trim
        # --- diagnostics: force_eager = never capture, run the SAME step body eagerly every step (RNG still drawn
        # outside in stock order) -> the stock function executed by this loop; traj = per-step x_l recorder {key: [tensor, ...]}
        # (force-eager = the disable_capture attribute: the same step body runs eagerly every step)
        self.traj: Optional[Dict[str, Any]] = None          # set to {"steps": [], "x0": None} by the worker to record x_l after every step (fp32 CPU copies)
        self.ec_guard = EmptyCacheGuard(self).install()      # EmptyCacheGuard: empty_cache deferred while >= 2 sampler graphs are live

    def _evict_oldest(self):
        """Drop the oldest entry: its graph (shared pool: release the family handle; private pool: the pool becomes freeable),
        its static input copies and its bias-cache buffers.  Numerics-free (a later prediction of that shape re-captures)."""
        k = self.order.pop(0)
        old = self.entries.pop(k, None)
        if old is None:
            return
        if old.get("graph") is not None:
            old["graph"] = None
            if self.pool_mode != "private":
                POOLS.release(self.family)
        self.stats["evictions"] += 1
        self.stats["evicted_bytes"] += int(old.get("bytes", 0))
        old.clear()   # drop st / cond / bc / body references so the static buffers are freed now, not at the next gc

    # ----- the deterministic step body (== stock math with the random numbers as inputs)
    def _drop_entry(self, key):
        """Drop one entry by key (lever sampler_prep[keycheck]: same shapes, different index values): the cleanup _evict_oldest does."""
        if key in self.order:
            self.order.remove(key)
        old = self.entries.pop(key, None)
        if old is None:
            return
        if old.get("graph") is not None:
            old["graph"] = None
            if self.pool_mode != "private":
                POOLS.release(self.family)
        self.stats["evictions"] += 1
        self.stats["evicted_bytes"] += int(old.get("bytes", 0))
        old.clear()

    @staticmethod
    def _step_body(denoise_net, st, dtype, batch_shape, chunk_n_sample, N_atom, gamma_on, noise_scale_lambda, step_scale_eta,
                   input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l, attn_chunk_size, inplace_safe,
                   enable_efficient_fusion):
        from protenix.model.utils import expand_at_dim, rot_vec_mul

        x_l = st["x_l"]
        # --- centre_random_augmentation(x_l, N_sample=1) with the rotation/translation supplied (Alg. 19)
        x_c = x_l - torch.mean(input=x_l, dim=-2, keepdim=True)
        x_c = expand_at_dim(x_c, dim=-3, n=1)  # [..., cs, 1, N_atom, 3]
        rot = st["rot"]  # [*batch_shape, cs, 1, 3, 3] float32 (already on device, == uniform_random_rotation().to(device).reshape(...))
        x_aug = rot_vec_mul(r=expand_at_dim(rot, dim=-3, n=N_atom), t=x_c) + st["trans"][..., None, :]
        x_l2 = x_aug.squeeze(dim=-3).to(dtype)
        # --- noise injection (stock: x_noisy = x_l + noise_scale_lambda * delta_noise_level * randn(...))
        x_noisy = x_l2 + noise_scale_lambda * st["delta"] * st["eps"]
        # --- denoise
        t_hat = st["t_hat"].reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, chunk_n_sample).to(dtype)
        x_denoised = denoise_net(x_noisy=x_noisy, t_hat_noise_level=t_hat, input_feature_dict=input_feature_dict, s_inputs=s_inputs,
                                 s_trunk=s_trunk, z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l, chunk_size=attn_chunk_size,
                                 inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
        delta = (x_noisy - x_denoised) / t_hat[..., None, None]
        dt = st["dt"]
        x_new = x_noisy + step_scale_eta * dt[..., None, None] * delta
        x_l.copy_(x_new)
        return x_l

    def sample(self, denoise_net, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l, noise_schedule, N_sample=1,
               gamma0=0.8, gamma_min=1.0, noise_scale_lambda=1.003, step_scale_eta=1.5, diffusion_chunk_size=None,
               inplace_safe=False, attn_chunk_size=None, enable_efficient_fusion=False, guidance_configs=None, stock_fn=None):
        """Drop-in for protenix.model.generator.sample_diffusion.  Falls back to `stock_fn` when guidance is on."""
        from protenix.tfg import parse_tfg_config
        from protenix.model.utils import uniform_random_rotation

        tfg_cfg = parse_tfg_config(guidance_configs)
        if tfg_cfg.enable:
            self.stats["eager_steps"] += 1
            return stock_fn(denoise_net=denoise_net, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk,
                            z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l, noise_schedule=noise_schedule, N_sample=N_sample,
                            gamma0=gamma0, gamma_min=gamma_min, noise_scale_lambda=noise_scale_lambda, step_scale_eta=step_scale_eta,
                            diffusion_chunk_size=diffusion_chunk_size, inplace_safe=inplace_safe, attn_chunk_size=attn_chunk_size,
                            enable_efficient_fusion=enable_efficient_fusion, guidance_configs=guidance_configs)

        if self.max_tokens and int(s_inputs.shape[-2]) > self.max_tokens:
            # BYPASS: above the token threshold the stock sampler runs unchanged (no capture, no static buffers, no bias cache).
            self.stats["bypass"] += 1
            prev_mode = self.biascache.mode if self.biascache is not None else None
            if self.biascache is not None:
                self.biascache.set_mode("off")
            try:
                return stock_fn(denoise_net=denoise_net, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk,
                                z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l, noise_schedule=noise_schedule, N_sample=N_sample,
                                gamma0=gamma0, gamma_min=gamma_min, noise_scale_lambda=noise_scale_lambda, step_scale_eta=step_scale_eta,
                                diffusion_chunk_size=diffusion_chunk_size, inplace_safe=inplace_safe, attn_chunk_size=attn_chunk_size,
                                enable_efficient_fusion=enable_efficient_fusion, guidance_configs=guidance_configs)
            finally:
                if self.biascache is not None:
                    self.biascache.set_mode(prev_mode)

        N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
        _key_digest_cache: dict = {}   # scoped to this sample() call; shared across its possibly-several _chunk() calls
        batch_shape = s_inputs.shape[:-2]
        device = s_inputs.device
        dtype = s_inputs.dtype

        def _chunk(chunk_n_sample):
            # ---- init noise: stock call, stock order
            x0 = noise_schedule[0] * torch.randn(size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype)
            # ---- per-step scalars exactly as stock (tensor ops on the schedule); the gamma pattern is part of the signature
            steps = []
            gamma_pattern = tuple(bool(v) for v in (noise_schedule[1:] > gamma_min).tolist())  # stock: one host compare per step
            if self.prep.stepvec:      # lever sampler_prep[stepvec]: the same fp32 elementwise arithmetic, three vector ops for all steps
                steps = _sp.step_scalars(noise_schedule, gamma_pattern, gamma0)
            else:
                for i, (c_tau_last, c_tau) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
                    gamma = float(gamma0) if gamma_pattern[i] else 0
                    t_hat = c_tau_last * (gamma + 1)
                    delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
                    dt = c_tau - t_hat
                    steps.append((t_hat, delta_noise_level, dt))
            key = ("v1", tuple(batch_shape), chunk_n_sample, N_atom, int(s_inputs.shape[-2]), str(dtype), str(s_inputs.dtype),
                   bool(inplace_safe), attn_chunk_size, bool(enable_efficient_fusion), float(noise_scale_lambda), float(step_scale_eta),
                   gamma_pattern, z_trunk is None, pair_z is None, p_lm is None, c_l is None,
                   (_sp.feature_shape_key(input_feature_dict) if self.prep.keycheck else _input_feature_key(input_feature_dict, _key_digest_cache)))
            ent = self.entries.get(key)
            if self.prep.keycheck:       # lever sampler_prep[keycheck]: integer/mask feature VALUES compared on the device with the live entry
                if ent is not None:
                    if _sp.same_index_values(ent, input_feature_dict):
                        _sp.STATS["key_hits"] += 1
                    else:
                        _sp.STATS["key_value_miss"] += 1; self._drop_entry(key); ent = None
                else:
                    _sp.STATS["key_shape_miss"] += 1
            cond = {"input_feature_dict": input_feature_dict, "s_inputs": s_inputs, "s_trunk": s_trunk, "z_trunk": z_trunk,
                    "pair_z": pair_z, "p_lm": p_lm, "c_l": c_l}
            N_augment = int(torch.numel(x0[..., 0, 0]))
            batch_size_shape = (*batch_shape, chunk_n_sample)

            def draw(i):
                """stock random draws of step i, in stock order: rotation (numpy/scipy) -> translation (torch.randn) -> eps."""
                rot = _sp.rotation_to_device(uniform_random_rotation(N_sample=N_augment), device, self.prep).reshape(*batch_size_shape, 1, 3, 3).detach()   # lever sampler_prep[rot_async]
                trans = 1.0 * torch.randn(size=(*batch_size_shape, 1, 3), device=device)
                eps = torch.randn(size=x0.shape, device=device, dtype=dtype)
                return rot, trans, eps

            if ent is None:
                _chain_env = None
                if self.prep.pool_chain and self.order and len(self.order) >= self.max_entries:   # lever sampler_prep[pool_chain]
                    self._prep_chain, _chain_env = _sp.prepare_capture(self, int(s_inputs.shape[-2]), int(N_atom), int(chunk_n_sample))
                ent = self._capture(key, denoise_net, cond, x0, dtype, batch_shape, chunk_n_sample, N_atom, steps[0], draw(0),
                                    gamma_pattern[0], noise_scale_lambda, step_scale_eta, attn_chunk_size, inplace_safe,
                                    enable_efficient_fusion)
                if self.prep.pool_chain and not ent.get("unsupported"):                        # lever sampler_prep[pool_chain]: what this entry's pool has served
                    ent["prep_envelope"] = _sp.envelope_after_capture(_chain_env, int(s_inputs.shape[-2]), int(N_atom), int(chunk_n_sample))
                st = ent["st"]
                start = 1  # the warm-up executed step 0 eagerly on the real inputs (x_l now holds x_1); its RNG draws were made
                self.stats["warmup_steps"] += 1
            else:
                st = ent["st"]
                copy_into(ent["cond"], cond)
                st["x_l"].copy_(x0)
                start = 0
                if self.biascache is not None:
                    self.biascache.bind(ent)
            g = ent.get("graph")
            body = ent["body"]
            for i in range(start, len(steps)):
                t_hat, dnl, dt = steps[i]
                rot, trans, eps = draw(i)  # stock draws, stock order, every step
                st["rot"].copy_(rot)
                st["trans"].copy_(trans)
                st["eps"].copy_(eps)
                st["t_hat"].copy_(t_hat)
                st["delta"].copy_(dnl)
                st["dt"].copy_(dt)
                if self.biascache is not None and i == 0:
                    # record step: step 0 executed eagerly with the STOCK ops; their outputs fill the static bias buffers this
                    # prediction's replays read (bias chain = pure function of pair_z; identical at every step)
                    self.biascache.set_mode("record")
                    with _autocast_nocache():
                        body()
                    self.biascache.set_mode("hit")
                    self.stats["record_steps"] += 1
                    continue
                if self.biascache is not None and g is not None and i in self.biascache_audit_steps:
                    self._biascache_audit_step(ent, i)
                if g is not None and i in self.teacher_forced_steps:
                    self._teacher_forced_step(ent, i, N_atom)
                if g is not None:
                    g.replay()
                    self.stats["replays"] += 1
                else:  # capture unsupported/disabled for this signature: the same step body, eager (numerically the stock step)
                    with _autocast_nocache():
                        body()
                    self.stats["eager_steps"] += 1
                if self.traj is not None:
                    if i == start and self.traj.get("x0") is None: self.traj["x0"] = x0.detach().float().cpu().clone()
                    self.traj["steps"].append((i, st["x_l"].detach().float().cpu().clone()))
                if i in self.nan_check_steps:
                    fin = bool(torch.isfinite(st["x_l"]).all().item())
                    self.stats["events"].append({"event": "nan_check", "step": i, "finite": fin, "x_abs_max": float(st["x_l"].abs().max().item())})
            return st["x_l"].clone()

        if diffusion_chunk_size is None:
            return _chunk(N_sample)
        outs = []
        no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
        for i in range(no_chunks):
            cs = diffusion_chunk_size if i < no_chunks - 1 else N_sample - i * diffusion_chunk_size
            outs.append(_chunk(cs))
        return torch.cat(outs, -3)

    def _biascache_audit_step(self, ent, i):
        """Eager step body in 'audit' mode from a snapshot: the stock LayerNorm / conv2d chain runs from scratch and every tensor is
        torch.equal-compared with the static buffer the captured graph reads (mismatch raises); the rollout then continues."""
        st = ent["st"]; body = ent["body"]
        x_snap = st["x_l"].clone(); torch.cuda.synchronize()
        self.biascache.set_mode("audit")
        try:
            with _autocast_nocache():
                body()
        finally:
            self.biascache.set_mode("hit")
        torch.cuda.synchronize(); st["x_l"].copy_(x_snap)
        self.stats["bc_audit_steps"] += 1

    def _poison_test(self, ent, N_atom):
        """Replay step 0 twice from one snapshot, the second time with the static biases overwritten: a large difference proves the
        graph consumes the static buffers (a graph reading a stale private copy would be unaffected)."""
        st = ent["st"]; g = ent["graph"]
        x_snap = st["x_l"].clone(); torch.cuda.synchronize()
        g.replay(); torch.cuda.synchronize(); x_a = st["x_l"].clone(); st["x_l"].copy_(x_snap)
        backup = self.biascache.poison(8.0)
        g.replay(); torch.cuda.synchronize(); x_b = st["x_l"].clone(); st["x_l"].copy_(x_snap)
        self.biascache.unpoison(backup); torch.cuda.synchronize()
        g.replay(); torch.cuda.synchronize(); x_c = st["x_l"].clone(); st["x_l"].copy_(x_snap)
        d_ab = float((x_a.float() - x_b.float()).abs().max()); d_ac = float((x_a.float() - x_c.float()).abs().max())
        if self.prep.poison_once:      # lever sampler_prep[poison_once]: a stale-copy graph gives poisoned ~ replay noise (ratio ~1); replay-to-replay noise is not 0 outside the deterministic recipe
            reads = bool(d_ab >= _sp.POISON_MIN_RATIO * max(d_ac, 1e-6) and d_ab >= _sp.POISON_MIN_ABS)
            rec = {"N_atom": N_atom, "poisoned_vs_normal_max": d_ab, "normal_vs_normal_max": d_ac, "det0_replay_noise_max": d_ac,
                   "criterion": f"poisoned >= {_sp.POISON_MIN_RATIO:g} x replay noise and >= {_sp.POISON_MIN_ABS:g}", "reads_static_buffers": reads}
        else:
            rec = {"N_atom": N_atom, "poisoned_vs_normal_max": d_ab, "normal_vs_normal_max": d_ac, "reads_static_buffers": bool(d_ab > 100.0 * max(d_ac, 1e-6) and d_ab > 0.05)}
        self.stats["poison"].append(rec); ent["poison"] = rec
        if not rec["reads_static_buffers"]:
            raise RuntimeError(f"biascache poison test FAILED: the captured graph does not read the static bias buffers ({rec})")
        return rec


    def _poison_test_classes(self, ent, N_atom):
        """lever sampler_prep[poison_once]: one replay per CLASS of captured input with that class overwritten — first by NaN (the step output
        x_l turns non-finite iff a captured kernel reads the live buffer: binary, threshold-free), and, for a class that stays finite, once more by
        a large finite pattern (a consumer kernel that maps non-finite operands to finite results — clamped logits, masked max — still moves x_l
        by far more than the replay-to-replay noise when it reads the buffer).  A class is READ when either stage sees it.  Which classes MUST be
        read is decided from the live objects, not from lever names (sampler_prep.poison_classify): every per-step input; every hoist slot this
        process RECORDED at the eager step (its producer wrapper was entered, dit_hoist Slot.n_rec > 0) unless sampler_prep names it as consumed
        at record time; hoist kinds whose producers were never entered (another lever serves their consumer) do not exist as slots and are
        reported as subsumed.  Integer / mask features cannot hold NaN and are covered by keycheck (values compared per item)."""
        st = ent["st"]; g = ent["graph"]; scond = ent.get("cond") or {}
        x_snap = st["x_l"].clone(); torch.cuda.synchronize()
        members = {}                                                    # class -> [(tensor, resync_fn or None)]; a class = one KIND of input (sampler_prep.poison_class_key)
        recorded = {}                                                   # hoist class -> every member slot was recorded this process (Slot.n_rec > 0)
        def add(name, t, resync=None, n_rec=None):
            if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel():
                key = _sp.poison_class_key(name)
                members.setdefault(key, []).append((t, resync))
                if n_rec is not None:
                    recorded[key] = recorded.get(key, True) and n_rec > 0
        for k, v in st.items():
            if k != "x_l":
                add("st." + k, v)
        for k, v in scond.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    add("cond." + k + "." + kk, vv)
            else:
                add("cond." + k, v)
        hoist_present = []
        if self.biascache is not None and getattr(self.biascache, "cur", None):
            for name, slot in (self.biascache.cur.get("slots") or {}).items():
                hoist_present.append(str(name))
                add("hoist." + str(name), getattr(slot, "buf", None),
                    (lambda n=name, sl=slot: self.biascache._g1_resync(n, sl)) if hasattr(self.biascache, "_g1_resync") else None,
                    n_rec=int(getattr(slot, "n_rec", 1) or 0))
        # reference replay + replay-to-replay noise (index-add atomics are not bitwise outside the deterministic recipe)
        g.replay(); torch.cuda.synchronize(); x_ref = st["x_l"].clone(); st["x_l"].copy_(x_snap)
        g.replay(); torch.cuda.synchronize(); noise = float((st["x_l"].float() - x_ref.float()).abs().max()); st["x_l"].copy_(x_snap); torch.cuda.synchronize()
        read, squash, unread, moved = [], [], [], {}
        for cname, group in members.items():                            # one (or two) replays per class, every member of the class poisoned together
            backups = [t.clone() for t, _ in group]
            def restore():
                for (t, resync), b in zip(group, backups):
                    t.copy_(b)
                    if resync is not None:
                        resync()
                st["x_l"].copy_(x_snap); torch.cuda.synchronize()
            for t, resync in group:                                    # stage 1: NaN
                t.fill_(float("nan"))
                if resync is not None:
                    resync()
            g.replay(); torch.cuda.synchronize()
            nonfinite = bool((~torch.isfinite(st["x_l"])).any().item())
            restore()
            if nonfinite:
                read.append(cname); del backups; continue
            if not _sp.poison_needs_magnitude(cname, recorded=recorded, hoist_bound=self.biascache is not None):   # stage 2 only where the verdict depends on it
                unread.append(cname); del backups; continue
            for (t, resync), b in zip(group, backups):                 # stage 2: large finite pattern (deterministic; sign flips and an offset)
                t.copy_(b * -3.0 + 7.0)
                if resync is not None:
                    resync()
            g.replay(); torch.cuda.synchronize()
            xo = st["x_l"].float()
            d = float("inf") if not bool(torch.isfinite(xo).all().item()) else float((xo - x_ref.float()).abs().max())
            restore()
            moved[cname] = d
            (squash if _sp.poison_moved(d, noise) else unread).append(cname)
            del backups
        verdict = _sp.poison_classify(read=read, squash=squash, unread=unread, recorded=recorded, hoist_bound=self.biascache is not None,
                                      hoist_present=hoist_present)
        _sp.POISON_STATUS.update(ran=True, unlisted=verdict["unlisted"], subsumed=verdict["subsumed"], squash=list(squash), failed=bool(verdict["missing"]))
        rec = {"N_atom": N_atom, "mode": "per-class NaN, then magnitude", "classes": len(members), "read": read, "read_finite_squash": squash, "unread": unread,
               "replay_noise_max": noise, "moved_max": {k: (round(v, 3) if v != float("inf") else "nonfinite") for k, v in moved.items()},
               "unread_expected": {n: _sp.POISON_EXPECTED_UNREAD.get(n, "not listed") for n in unread},
               "subsumed": verdict["subsumed"], "unlisted": verdict["unlisted"], "required_missing": verdict["missing"], "reads_static_buffers": not verdict["missing"]}
        self.stats["poison"].append(rec); ent["poison_classes"] = rec
        line = _sp.poison_status()
        print(f"[infopt_graphs] poison per-class probe: {line} (classes={len(members)} read={len(read)} read_finite_squash={len(squash)} unread={len(unread)} "
              f"replay_noise_max={noise:.3g}" + (f"; squash={squash}" if squash else "") + (f"; subsumed={verdict['subsumed']}" if verdict["subsumed"] else "")
              + (f"; unlisted={verdict['unlisted']}" if verdict["unlisted"] else "") + ")", flush=True)
        if verdict["missing"]:
            raise RuntimeError(f"poison test FAILED: the captured graph does not read the live static buffers of {verdict['missing']} ({rec})")
        return rec
    def _teacher_forced_step(self, ent, i, N_atom):
        """Teacher-forced per-step parity on IDENTICAL inputs: eager body twice (run-to-run floor) vs graph replay, from the same
        static buffers; all three start from the same x_l snapshot; the rollout then continues from the graph result."""
        st = ent["st"]; body = ent["body"]; g = ent["graph"]
        x_snap = st["x_l"].clone()
        torch.cuda.synchronize()
        if self.biascache is not None:
            self.biascache.set_mode("audit")   # eager passes recompute the bias chain with the stock ops and torch.equal it against the static buffers
        try:
            with _autocast_nocache():
                body(); torch.cuda.synchronize(); x_e1 = st["x_l"].clone(); st["x_l"].copy_(x_snap)
                body(); torch.cuda.synchronize(); x_e2 = st["x_l"].clone(); st["x_l"].copy_(x_snap)
        finally:
            if self.biascache is not None:
                self.biascache.set_mode("hit")
        g.replay(); torch.cuda.synchronize(); x_g = st["x_l"].clone(); st["x_l"].copy_(x_snap)
        d_ge = (x_g.float() - x_e1.float()).abs(); d_ee = (x_e2.float() - x_e1.float()).abs()
        rec = {"step": i, "N_atom": N_atom, "d_graph_vs_eager_max": float(d_ge.max()), "d_graph_vs_eager_mean": float(d_ge.mean()),
               "d_eager_vs_eager_max": float(d_ee.max()), "d_eager_vs_eager_mean": float(d_ee.mean()),
               "graph_bitwise_equal_eager": bool(torch.equal(x_g, x_e1)), "eager_bitwise_repeatable": bool(torch.equal(x_e1, x_e2)),
               "x_abs_max": float(x_e1.float().abs().max())}
        self.teacher_forced.append(rec)
        return rec

    def _capture(self, key, denoise_net, cond, x0, dtype, batch_shape, chunk_n_sample, N_atom, step0, draw0, gamma_on,
                 noise_scale_lambda, step_scale_eta, attn_chunk_size, inplace_safe, enable_efficient_fusion):
        # With the DiT hoist bound (biascache), static bias buffers are (re)bound per capture and older graphs can
        # reference released buffers
        # -> more than ONE cached sampler graph is UNSUPPORTED with the hoist: clamp to 1.
        if getattr(self, "biascache", None) is not None and self.max_entries > 1:
            print(f"[infopt_graphs] HAZARD #44 policy: DiT hoist (biascache) is bound -> sampler graph cache clamped to 1 entry (was max_entries={self.max_entries})", flush=True)
            self.max_entries = 1
            while len(self.order) > 1:
                self._evict_oldest()
        t0 = time.time()
        scond = static_like(cond)
        copy_into(scond, cond)
        rot0, trans0, eps0 = draw0
        st = {"x_l": x0.clone(), "rot": rot0.clone(), "trans": trans0.clone(), "eps": eps0.clone(),
              "t_hat": step0[0].clone(), "delta": step0[1].clone(), "dt": step0[2].clone()}
        torch.cuda.synchronize()
        body = lambda: self._step_body(denoise_net, st, dtype, batch_shape, chunk_n_sample, N_atom, gamma_on, noise_scale_lambda,
                                       step_scale_eta, scond["input_feature_dict"], scond["s_inputs"], scond["s_trunk"], scond["z_trunk"],
                                       scond["pair_z"], scond["p_lm"], scond["c_l"], attn_chunk_size, inplace_safe, enable_efficient_fusion)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        audit = None
        ent = {"key": key, "st": st, "cond": scond, "gamma": gamma_on, "body": body}
        if self.biascache is not None:
            self.biascache.bind(ent)
        # warm-up = step 0 executed eagerly on the real inputs (a side stream, as in the torch docs pattern); the sync census
        # runs in WARN mode here (a first call may legitimately sync for lazy inits: Triton compile, cuDNN plan cache).
        x0_keep = x0.clone()
        audit = {}
        with torch.cuda.stream(s), _autocast_nocache():
            for pas in range(1 if self.prep.warmup1 else 2):     # lever sampler_prep[warmup1]
                st["x_l"].copy_(x0_keep)
                if self.biascache is not None:
                    self.biascache.set_mode("record" if pas == 0 else "hit")   # pass 1 fills the static buffers with the stock ops' outputs
                with RNGGuard(raise_on_use=False) as rg, SyncCensus(mode="warn") as sc:
                    body()  # step 0 on the real inputs (pass 2 recomputes the identical step; x_l == x_1 afterwards)
                audit[f"pass{pas + 1}"] = {"syncs": sc.report(), "rng_consumed": rg.consumed}
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        if self.biascache is not None:
            self.biascache.set_mode("hit")   # the capture records the step reading the static buffers
        if rg.consumed and self.rng_guard:
            ent["unsupported"] = f"RNGGuard: step body consumed RNG {rg.consumed}"
            self.stats["events"].append({"event": "sampler_rng_guard", "error": ent["unsupported"]})
            self.entries[key] = ent
            return ent
        warm_s = time.time() - t0
        if self.disable_capture:
            ent["audit"] = audit; ent["warmup_s"] = warm_s; ent["graph"] = None; ent["unsupported"] = "capture disabled (disable_capture)"
            self.entries[key] = ent; self.order.append(key)
            self.stats["events"].append({"event": "sampler_capture_disabled", "N_atom": N_atom, "audit": audit})
            return ent
        t1 = time.time()
        reserved0 = torch.cuda.memory_reserved()
        if self.prep.pool_chain and self._prep_chain is not None:      # lever sampler_prep[pool_chain]: the previous graph's pool (that graph is alive, never replayed again)
            pool = self._prep_chain.pool()
        else:
            pool = None if self.pool_mode == "private" else POOLS.acquire(self.family)
        g = torch.cuda.CUDAGraph()
        try:
            with _autocast_nocache(), capture_context(g, pool=pool, stream=s):
                body()
        except Exception as ex:  # noqa
            if is_oom(ex): raise                                   # an out-of-memory is the caller's to see: never rerouted (opt_core.oom)
            torch.cuda.synchronize()
            if pool is not None:
                POOLS.release(self.family)
            ent["unsupported"] = f"{type(ex).__name__}: {str(ex)[:400]}"
            ent["audit"] = audit
            self.stats["events"].append({"event": "sampler_capture_failed", "error": ent["unsupported"], "audit": audit})
            self._prep_chain = None
            self.entries[key] = ent
            return ent
        torch.cuda.synchronize()
        ent["graph"] = g
        ent["capture_s"] = time.time() - t1
        if self.biascache is not None and self.poison_check and self.prep.poison_once and self.stats.get("poison"):
            _sp.STATS["poison_skipped"] += 1                             # lever sampler_prep[poison_once]: shown once in this process
        elif self.poison_check and self.prep.poison_once:               # lever sampler_prep[poison_once]: first capture of the process — per-class non-finite probe (+ the magnitude probe when the hoist is bound)
            x1_keep = st["x_l"].clone()
            if self.biascache is not None:
                self._poison_test(ent, N_atom)
            self._poison_test_classes(ent, N_atom)
            st["x_l"].copy_(x1_keep); torch.cuda.synchronize()
        elif self.biascache is not None and self.poison_check:
            x1_keep = st["x_l"].clone()
            self._poison_test(ent, N_atom)
            st["x_l"].copy_(x1_keep); torch.cuda.synchronize()
        ent["warmup_s"] = warm_s
        ent["pool_growth_mb"] = (torch.cuda.memory_reserved() - reserved0) / 1e6
        ent["static_bytes"] = _static_bytes(ent)
        ent["bytes"] = int(max(ent["pool_growth_mb"], 0.0) * 1e6) + ent["static_bytes"]
        ent["audit"] = audit
        self.entries[key] = ent
        self.order.append(key)
        n_evicted = 0
        while len(self.order) > self.max_entries:
            self._evict_oldest(); n_evicted += 1
        if self.max_pool_bytes:
            while len(self.order) > 1 and sum(self.entries[k].get("bytes", 0) for k in self.order) > self.max_pool_bytes:
                self._evict_oldest(); n_evicted += 1
        self._prep_chain = None                                          # lever sampler_prep[pool_chain]: the previous graph is dropped here; the empty_cache below trims cached blocks only
        if (n_evicted or self._prep_renewed) and self.pool_mode == "private" and self.evict_empty_cache:   # lever sampler_prep[pool_chain]: also after a renewed capture (its eviction ran before the capture)
            self._prep_renewed = False
            torch.cuda.synchronize(); torch.cuda.empty_cache()   # a dropped private pool is freeable: return it to the device now
        self.ec_guard.flush_if_safe()
        ent["n_evicted_at_capture"] = n_evicted
        self.stats["captures"] += 1
        self.stats["events"].append({"event": "sampler_captured", "N_atom": N_atom, "N_token": key[4], "chunk_n_sample": chunk_n_sample,
                                     "warmup_s": round(warm_s, 4), "capture_s": round(ent["capture_s"], 4),
                                     "pool_growth_mb": round(ent["pool_growth_mb"], 1), "audit": audit})
        return ent

    def clear(self):
        for k in list(self.entries):
            old = self.entries.pop(k)
            if old.get("graph") is not None and self.pool_mode != "private":
                old["graph"] = None
                POOLS.release(self.family)
        self.order.clear()

    def prep_report(self):
        """lever sampler_prep: {"on", "parts" (word), "poison" (ok|unlisted:…|failed|-), "stats" (sampler_prep.STATS)} for the kit's LEVER line."""
        return {"on": bool(self.prep.on), "parts": self.prep.describe(), "poison": _sp.poison_status(), "stats": _sp.report()}

    def summary(self):
        return {"captures": self.stats["captures"], "replays": self.stats["replays"], "warmup_steps": self.stats["warmup_steps"],
                "eager_steps": self.stats["eager_steps"], "entries": len(self.entries),
                "capture_s": [round(e.get("capture_s", 0), 4) for e in self.entries.values()],
                "warmup_s": [round(e.get("warmup_s", 0), 4) for e in self.entries.values()],
                "pool_growth_mb": [round(e.get("pool_growth_mb", 0), 1) for e in self.entries.values()],
                "entry_bytes_gb": [round(e.get("bytes", 0) / 1e9, 3) for e in self.entries.values()],
                "bypass": self.stats["bypass"], "evictions": self.stats["evictions"], "evicted_bytes_gb": round(self.stats["evicted_bytes"] / 1e9, 3),
                "max_pool_bytes": self.max_pool_bytes, "max_tokens": self.max_tokens, "pool_mode": self.pool_mode, "max_entries": self.max_entries, "empty_cache_guard": self.ec_guard.report(),
                "teacher_forced": self.teacher_forced[-40:], "record_steps": self.stats["record_steps"], "bc_audit_steps": self.stats["bc_audit_steps"],
                "poison": self.stats["poison"][-40:], "biascache": (self.biascache.summary() if self.biascache is not None else None),
                "events": self.stats["events"][-20:]}


# ---------------------------------------------------------------------------------------------------------------------
# (b) graphed trunk body (PairformerStack per recycle)
# ---------------------------------------------------------------------------------------------------------------------


def _graph_pairformer_stack(stack: torch.nn.Module, family: str, name: str, pool: str, rng_guard: bool, sync_audit: bool,
                            max_entries: int) -> GraphedFunction:
    orig_forward = stack.forward

    def body(s, z, tm, ta, inplace_safe):
        return orig_forward(s, z, pair_mask=None, triangle_multiplicative=tm, triangle_attention=ta, inplace_safe=inplace_safe,
                            chunk_size=None)

    gf = GraphedFunction(body, name=name, family=family, pool=pool, clone_outputs=True, max_entries=max_entries, rng_guard=rng_guard,
                         sync_audit=sync_audit)

    def forward(self, s, z, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False,
                chunk_size=None):
        if self.training or pair_mask is not None or chunk_size is not None or torch.is_grad_enabled():
            gf.stats["eager_disabled"] += 1
            return orig_forward(s, z, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative,
                                triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        return gf(s, z, triangle_multiplicative, triangle_attention, bool(inplace_safe))

    stack.forward = types.MethodType(forward, stack)
    stack._infopt_graphed = gf
    stack._infopt_orig_forward = orig_forward
    return gf


# ---------------------------------------------------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------------------------------------------------


FASTLN_REPORT: Dict[str, Any] = {}


def install(model, sampler: bool = True, trunk: bool = False, confidence_trunk: bool = False, pool: str = "shared",
            rng_guard: bool = True, sync_audit: bool = True, max_entries: int = 8, fastln_stream_fix: bool = True,
            max_pool_bytes: int = 0, max_tokens: int = 0, sampler_prep=False) -> Dict[str, Any]:
    """Patch a constructed protenix.model.protenix.Protenix instance.  Returns handles with .summary().
    trunk=False by default (exact; no kit mode turns it on).
    fastln_stream_fix: rebuild Protenix's fast_layernorm extension with stream-correct launches (REQUIRED for capture: the
    stock extension launches on the legacy default stream and is therefore not recorded by a graph -> NaN on replay)."""
    handles: Dict[str, Any] = {}
    if fastln_stream_fix and os.environ.get("LAYERNORM_TYPE", "fast_layernorm") == "fast_layernorm":
        from .fastln_stream import install_stream_correct_fastln
        FASTLN_REPORT.update(install_stream_correct_fastln())
        handles["fastln_stream"] = FASTLN_REPORT
    if sampler:
        import protenix.model.protenix as PM
        from protenix.model import generator as G
        max_pool_bytes = int(float(max_pool_bytes / 1e9 if max_pool_bytes else 0) * 1e9)
        loop = GraphedDenoiseLoop(family="protenix_sampler", pool=pool, rng_guard=rng_guard, sync_audit=sync_audit, max_entries=max_entries,
                                  max_pool_bytes=max_pool_bytes, max_tokens=max_tokens, prep=_sp.PrepConfig(bool(sampler_prep)))   # lever sampler_prep
        stock = G.sample_diffusion

        def graphed_sample_diffusion(**kw):
            return loop.sample(stock_fn=stock, **kw)

        # Protenix.sample_diffusion resolves `sample_diffusion` from its module globals at call time -> patch the name there.
        PM.sample_diffusion = graphed_sample_diffusion
        model._infopt_sampler = loop
        handles["sampler"] = loop
    if trunk:
        handles["trunk"] = _graph_pairformer_stack(model.pairformer_stack, "protenix_trunk", "pairformer_stack", pool, rng_guard,
                                                   sync_audit, max_entries)
    if confidence_trunk and hasattr(model, "confidence_head") and hasattr(model.confidence_head, "pairformer_stack"):
        handles["confidence_trunk"] = _graph_pairformer_stack(model.confidence_head.pairformer_stack, "protenix_trunk",
                                                              "confidence_pairformer_stack", pool, rng_guard, sync_audit, max_entries)
    model._infopt_handles = handles
    return handles


def uninstall(model):
    """Restore the stock sampler/trunk; drop all graphs and release their pool shares (the fast-LN stream fix stays: it is
    bit-equal to the original and needed by nothing else)."""
    import protenix.model.protenix as PM
    from protenix.model import generator as G
    PM.sample_diffusion = G.sample_diffusion
    h = getattr(model, "_infopt_handles", {})
    if "sampler" in h:
        try:
            getattr(h["sampler"], "ec_guard", None) and h["sampler"].ec_guard.uninstall()   # restore torch.cuda.empty_cache, flush a pending deferred release
        except Exception:
            pass
        h["sampler"].clear()
    for m in [model.pairformer_stack, getattr(getattr(model, "confidence_head", None), "pairformer_stack", None)]:
        if m is not None and hasattr(m, "_infopt_orig_forward"):
            m.forward = m._infopt_orig_forward
            m._infopt_graphed.clear()
            del m._infopt_orig_forward
            del m._infopt_graphed
    model._infopt_handles = {}
    torch.cuda.synchronize()


def summary(model) -> Dict[str, Any]:
    out = {}
    for k, h in getattr(model, "_infopt_handles", {}).items():
        out[k] = h.summary() if hasattr(h, "summary") else {kk: vv for kk, vv in h.items() if kk in ("installed", "build_s", "patched_launches", "reason")} | {
            "bitwise_all_equal": h.get("bitwise", {}).get("all_bitwise_equal"), "side_stream": h.get("side_stream")}
    return out
