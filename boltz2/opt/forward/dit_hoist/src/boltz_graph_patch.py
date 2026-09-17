"""boltz_graph_patch.py — CUDA-graph capture of the Boltz-2 diffusion sampling step (boltz 2.2.1, inference only).

Tier 1 (exact): the computed function is unchanged — same kernels in the
same order, same numerics, same checkpoint, same precision, same number of steps (200), same CUDA-RNG consumption.

When enabled it replaces `AtomDiffusion.sample` (boltz/model/modules/diffusionv2.py) by `sample_graphed`, which
  (1) evaluates the sigma/gamma schedule on the host ONCE (one .tolist() sync instead of 3 .item() syncs per step; the values are the
      same float32 numbers the stock loop reads via .item(), so t_hat / noise_var / step coefficients are the same Python floats),
  (2) pre-draws ALL random tensors of the loop on the CUDA generator in EXACTLY the stock order
      (init noise; then per step: random_quaternions randn((m,4)), translation randn((m,1,3)), eps randn(shape)),
      so the CUDA generator state after sampling is identical to stock and every downstream draw is unchanged
      (Philox offsets depend only on numel/device, not on when the call is made),
  (3) captures ONE "denoiser step" = [centre + random augmentation of x_t and of x0_prev, noise add, chunked preconditioned
      DiffusionModule forward] into a torch.cuda.CUDAGraph with static input/output buffers and replays it for steps 1..S-1
      (step 0 runs eagerly: in stock its x0_prev-augmentation branch is absent, so step 0 has a different op list),
  (4) keeps weighted_rigid_align and the Euler update eager between replays. The alignment cannot be captured (host-side guards
      `torch.any(...)` / `(S.abs() <= 1e-15).any()` and the cuSOLVER gesvd error check inside torch.linalg.svd are stream syncs);
      the Euler update is kept eager ON PURPOSE: its division by the Python float t_hat is computed by CUDA as
      x * (1.0f/float(t_hat)) and its step coefficient is a Python float cast to fp32 inside the kernel — keeping the literal stock
      expression (Python scalars) guarantees bit-identical arithmetic, at the cost of ~4 tiny launches per step.
The stock code path is untouched when the flag is off (the original method is kept as AtomDiffusion._stock_sample).

Flags
  BOLTZ_GRAPH_DIFFUSION = 0 | off      -> nothing patched (default)
                        = 1 | graph    -> pre-draw + hoisted schedule + CUDA graph
                        = predraw      -> pre-draw + hoisted schedule only, eager network step (isolates the sync-removal effect)
  BOLTZ_GRAPH_DEBUG=1                  -> verbose prints (capture time, pool size, key misses)
  BOLTZ_GRAPH_CAPTURE_MODE             -> 'thread_local' (default; the DataLoader pin-memory thread may touch CUDA) | 'global' | 'relaxed'
  BOLTZ_GRAPH_WS_IN_POOL=1 (default)   -> the cuBLAS workspace of the capture stream is (re)allocated INSIDE the capture, i.e. in the graph's
                                          private pool (lifetime = the graph); =0 keeps the warm-up's workspace (regular allocator), which
                                          Lightning's per-predict teardown (clearCublasWorkspaces + empty_cache) can free under the graph
                                          -> 'CUDA error: an illegal memory access' on the first replay after a recapture
  BOLTZ_GRAPH_KEEP_CUBLAS_WS=1 (default) -> in graph mode, torch._C._cuda_clearCublasWorkspaces() is additionally made a no-op for the
                                          process (belt and braces; scratch memory only, no numerics); =0 restores the stock call
  BOLTZ_GRAPH_RELEASE_MIN_TOKENS=1024 (default) -> lifetime of the captured step graph (its private pool and static conditioning clones) and,
                                          with the hoist composed, of the hoist's cache: an input below this many tokens keeps them for the
                                          process (later sample() calls of the same shapes refresh the statics and replay — no re-capture);
                                          an input at or above it releases them when its sample() returns (release_step_graph), so the
                                          confidence module and the next prediction's trunk run with the stock sampler's resident memory,
                                          and the next sample() re-captures (3 warm-up steps + 1 capture). The kept state grows with
                                          N_tokens^2 (the hoist's level-2 cache ~3 KB x N^2, the conditioning clones, the pool); =0 keeps
                                          it for the process at every size. Numerics unchanged either way (a re-capture is the first
                                          capture's arithmetic).
  BOLTZ_GRAPH_HEADROOM_MARGIN=0.12 (default) -> the memory-headroom gate of the sampler levers (headroom; applied per sample() by the hoist,
                                          boltz_dit_hoist.sample_hoisted, which composes this patch on every shipped row): before any of their
                                          state is built, the levers' projected working set for this input — the hoist's cache
                                          (boltz_dit_hoist.projected_cache_bytes), this patch's static clones (projected_static_bytes), the eager
                                          step's transients twice (once in the caching allocator at step 0, once as the graph's private pool:
                                          step_transient_bytes) — plus this fraction of the card must fit in the memory free at that moment;
                                          when it does not, that prediction samples on the stock path (eager, no hoist, no graph: the arithmetic
                                          the graph replays, so its outputs are the same bytes) and is counted `headroom_gated`. The levers thus
                                          fit wherever the stock sampler fits, by their own rule, at every input size and card.
Programmatic: `import boltz_graph_patch as BGP; BGP.apply("graph")` / `BGP.apply("off")`.
Stock CLI: `PYTHONPATH=<dir with this file and sitecustomize.py> BOLTZ_GRAPH_DIFFUSION=1 boltz predict ...`.

Fallback: if the capture raises (an uncapturable op inside DiffusionModule), the traceback is stored in STATS["capture_error"], printed once,
and the sampler continues in 'predraw' mode for this and all later calls, so a prediction never fails because of this patch.
Sampling with steering (fk_steering / physical or contact guidance) or in training mode uses the stock sampler unchanged.
"""
from __future__ import annotations

import functools
import os
import sys
import time
import traceback
from math import sqrt

import torch
from opt_core.oom import is_oom          # the core's one out-of-memory predicate: an out-of-memory error propagates, no fallback applied

STATS = {"mode": "off", "captures": 0, "capture_s": [], "capture_pool_gb": [], "replays": 0, "predraw_calls": 0, "graph_calls": 0,
         "stock_calls": 0, "capture_error": None, "sampler_s": [], "releases": 0, "headroom_gated": 0, "last": {}}
_DEBUG = os.environ.get("BOLTZ_GRAPH_DEBUG", "0") == "1"
RELEASE_MIN_TOKENS = int(os.environ.get("BOLTZ_GRAPH_RELEASE_MIN_TOKENS", "1024") or 0)   # see the flag above; 0 = keep for the process at every size


HEADROOM_MARGIN_FRAC = float(os.environ.get("BOLTZ_GRAPH_HEADROOM_MARGIN", "0.12"))   # see the flag above: the fraction of the card kept free beyond the projection
GIB = float(2 ** 30)


def step_transient_bytes(nck, multiplicity, heads=16) -> int:
    """The eager denoiser step's own high-water above its inputs, by shape: three fp32 ``[B*m, heads, N, N]`` temporaries of one token-transformer
    layer (pair bias, attention logits, attention weights) — the term that grows with the input; the graph's private pool holds the same."""
    s_trunk = nck["s_trunk"]; b, n = int(s_trunk.shape[0]), int(s_trunk.shape[1])
    return 3 * b * int(multiplicity) * int(heads) * n * n * 4


def projected_static_bytes(nck, multiplicity, n_atoms) -> int:
    """Bytes _get_step_graph clones for a capture of these inputs: s_trunk, s_inputs, every conditioning tensor, and the five coordinate-shaped
    fp32 buffers (x, den_prev, den_aug, noisy, den: ``[B*m, n_atoms, 3]``); the feats it clones (the keys the network reads) are token/atom-linear
    and left to the margin."""
    dc = nck["diffusion_conditioning"]; b = int(nck["s_trunk"].shape[0])
    n = sum(int(v.numel()) * int(v.element_size()) for v in (nck["s_trunk"], nck["s_inputs"]))
    n += sum(int(v.numel()) * int(v.element_size()) for v in dc.values() if torch.is_tensor(v))
    return n + 5 * b * int(multiplicity) * int(n_atoms) * 3 * 4


def headroom(projected_bytes, transient_bytes) -> dict:
    """The memory-headroom gate's decision for one sample(): ``gated`` when the projected working set (``projected_bytes``: the hoist cache + the
    static clones), the step transients twice and HEADROOM_MARGIN_FRAC of the card do not fit in what is free now (the driver's free memory plus the
    caching allocator's unused reserve). GiB figures for the census."""
    free_b, total_b = torch.cuda.mem_get_info()
    reusable = max(0, int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated()))
    free = int(free_b) + reusable; margin = HEADROOM_MARGIN_FRAC * float(total_b)
    need = float(projected_bytes) + 2.0 * float(transient_bytes) + margin
    r = lambda x: round(float(x) / GIB, 2)  # noqa: E731
    return {"gated": need > free, "projected_gib": r(projected_bytes), "transient_gib": r(transient_bytes), "margin_gib": r(margin),
            "need_gib": r(need), "free_gib": r(free), "total_gib": r(total_b)}


def note_headroom_gated(gate: dict, n_tokens, multiplicity, steps=None) -> None:
    """Record a sample() the gate sent to the stock path: the per-call census the worker copies per item (sampler_mode eager_headroom, no replay,
    no capture), the process count."""
    STATS["headroom_gated"] += 1; STATS["stock_calls"] += 1
    STATS["last"] = {"sampler_mode": "eager_headroom", "steps": steps, "multiplicity": multiplicity, "n_tokens": int(n_tokens), "n_replay": 0, "n_capture": 0,
                     "captures_total": STATS["captures"], "capture_s_last": None, "capture_pool_gb_last": None, "release_min_tokens": RELEASE_MIN_TOKENS,
                     "released": False, "headroom_gated": True, **{k: gate.get(k) for k in ("projected_gib", "transient_gib", "margin_gib", "need_gib", "free_gib")}}
    _log("memory headroom: need %.1f GiB (levers %.1f + step transients 2x%.1f + margin %.1f) > free %.1f GiB -> this prediction samples on the stock path"
         % (gate.get("need_gib") or 0, gate.get("projected_gib") or 0, gate.get("transient_gib") or 0, gate.get("margin_gib") or 0, gate.get("free_gib") or 0))


def release_due(n_tokens) -> bool:
    """Whether a sample() of an input of `n_tokens` (padded token count) releases the step graph and the hoist's cache when it returns."""
    return RELEASE_MIN_TOKENS > 0 and int(n_tokens) >= RELEASE_MIN_TOKENS


def release_step_graph(diff) -> bool:
    """Drop `diff`'s captured step graph — the CUDAGraph, its private memory pool, the static clones of the conditioning tensors — and return
    the freed blocks to the driver, so nothing of the sampler outlives its sample(). The cuBLAS workspace map entries that point into the pool
    are cleared first (the next GEMM allocates a fresh workspace; scratch memory, no numerics). True when a graph was dropped."""
    G = getattr(diff, "_step_graph", None)
    if G is None:
        return False
    diff._step_graph = None
    del G
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        _clear_cublas_workspaces_now()
        torch.cuda.empty_cache()
    STATS["releases"] += 1
    _log("released the step graph (input at or above BOLTZ_GRAPH_RELEASE_MIN_TOKENS=%d)" % RELEASE_MIN_TOKENS)
    return True


def _log(*a):
    if _DEBUG:
        print("[boltz_graph_patch]", *a, file=sys.stderr, flush=True)


class _RecordingDict(dict):
    """feats wrapper that records which keys the network reads (used at warm-up to know which feats tensors need static buffers)."""

    def __init__(self, d):
        super().__init__(d)
        self.accessed = set()

    def __getitem__(self, k):
        self.accessed.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.accessed.add(k)
        return super().get(k, default)


def _sig(t):
    return (tuple(t.shape), str(t.dtype), t.device.index, tuple(t.stride()))


def _stock_precond_forward(self, noised_atom_coords, sigma, network_condition_kwargs):
    """== AtomDiffusion.preconditioned_network_forward with `sigma` already a (batch,) fp32 tensor (stock builds the same tensor
    with torch.full((batch,), t_hat)); every other line is the stock code, calling the stock c_in/c_noise/c_skip/c_out."""
    from einops import rearrange
    padded_sigma = rearrange(sigma, "b -> b 1 1")
    r_update = self.score_model(r_noisy=self.c_in(padded_sigma) * noised_atom_coords, times=self.c_noise(sigma), **network_condition_kwargs)
    return self.c_skip(padded_sigma) * noised_atom_coords + self.c_out(padded_sigma) * r_update


class _StepGraph:
    """Static buffers + captured CUDA graph of one denoiser step for a fixed key (shapes/strides/dtypes, chunking, parameter pointers)."""

    def __init__(self, diff, key):
        self.diff = diff
        self.key = key
        self.graph = None
        self.static = {}
        self.feats_keys = None
        self.capture_s = None
        self.pool_gb = None

    def cond_kwargs(self):
        S = self.static
        return dict(s_trunk=S["s_trunk"], s_inputs=S["s_inputs"], feats=S["feats"], diffusion_conditioning=S["dc"])

    def body(self):
        """Stock lines 351-396 of AtomDiffusion.sample for one step i >= 1, with the random tensors and t_hat read from static buffers.
        inputs : x (atom_coords after the previous Euler update), den_prev (previous network output), R, tr, eps, that
        outputs: noisy (x after augmentation + noise), den (network output)"""
        S = self.static
        x, den_prev, R, tr, eps, that = S["x"], S["den_prev"], S["R"], S["tr"], S["eps"], S["that"]
        atom_coords = x - x.mean(dim=-2, keepdims=True)                                         # stock 354
        atom_coords = torch.einsum("bmd,bds->bms", atom_coords, R) + tr                          # stock 355-357
        atom_coords_denoised = den_prev - den_prev.mean(dim=-2, keepdims=True)                  # stock 359 (in-place in stock; same kernel math)
        atom_coords_denoised = torch.einsum("bmd,bds->bms", atom_coords_denoised, R) + tr       # stock 360-363 (dead value in stock too)
        S["den_aug"].copy_(atom_coords_denoised)
        if S["sgu"] is not None:
            S["sgu"].copy_(torch.einsum("bmd,bds->bms", S["sgu"], R))                          # stock 364-370 (guidance accumulator rotation)
        atom_coords_noisy = atom_coords + S["sqrt_nv"] * eps                                    # stock 377-378: eps = sqrt(noise_var)*randn
        atom_coords_denoised = torch.zeros_like(atom_coords_noisy)                              # stock 381
        for ids in S["chunks"]:                                                                 # stock 382-396
            sigma = S["sigma_%d" % ids.numel()]
            sigma.copy_(that.expand(ids.numel()))                                               # == torch.full((batch,), t_hat)
            chunk = self.diff._graph_precond_forward(atom_coords_noisy[ids], sigma, dict(multiplicity=ids.numel(), **self.cond_kwargs()))
            atom_coords_denoised[ids] = chunk
        S["noisy"].copy_(atom_coords_noisy)
        S["den"].copy_(atom_coords_denoised)


def _get_step_graph(self, x, den_prev, R, tr, eps, chunks, nck, sgu=None):
    """Return a _StepGraph valid for the current shapes/parameters, (re)capturing if needed. None on capture failure (logged once)."""
    dc = nck["diffusion_conditioning"]; feats = nck["feats"]
    param_key = hash(tuple(p.data_ptr() for p in self.score_model.parameters()))
    key = (_sig(x), _sig(den_prev), _sig(R), _sig(tr), _sig(eps), tuple(int(c.numel()) for c in chunks), _sig(nck["s_trunk"]), (sgu is not None),
           _sig(nck["s_inputs"]), tuple((k, _sig(v)) for k, v in sorted(dc.items()) if torch.is_tensor(v)), param_key,
           torch.is_autocast_enabled(), torch.get_float32_matmul_precision())
    G = getattr(self, "_step_graph", None)
    if G is not None and G.key == key:
        st = G.static                                    # same shapes: refresh the per-prediction conditioning values only
        st["s_trunk"].copy_(nck["s_trunk"]); st["s_inputs"].copy_(nck["s_inputs"])
        for k, v in dc.items():
            if torch.is_tensor(v):
                st["dc"][k].copy_(v)
            elif isinstance(v, functools.partial) and "indexing_matrix" in v.keywords:
                st["dc"][k].keywords["indexing_matrix"].copy_(v.keywords["indexing_matrix"])
        for k in feats:
            if k in G.feats_keys:
                st["feats"][k].copy_(feats[k])
            else:
                st["feats"][k] = feats[k]                  # never read by the network; keep the current object (no stale refs)
        return G
    if G is not None:
        _log("graph key miss -> recapture")
        self._step_graph = None; del G; torch.cuda.synchronize()
        _clear_cublas_workspaces_now()          # drop map entries that pointed into the deleted graph's pool (never dereference them again)
    G = _StepGraph(self, key)
    try:
        t0 = time.time()
        st = G.static
        st["x"] = x.clone(); st["den_prev"] = den_prev.clone(); st["den_aug"] = torch.empty_like(den_prev)
        st["noisy"] = torch.empty_like(x); st["den"] = torch.empty_like(x)
        st["R"] = R.clone(); st["tr"] = tr.clone(); st["eps"] = eps.clone()
        st["sgu"] = sgu.clone() if sgu is not None else None
        st["that"] = torch.zeros((), dtype=torch.float32, device=x.device); st["sqrt_nv"] = torch.zeros((), dtype=torch.float32, device=x.device)
        st["chunks"] = [c.clone() for c in chunks]
        for c in chunks:
            st["sigma_%d" % c.numel()] = torch.empty((c.numel(),), dtype=torch.float32, device=x.device)
        st["s_trunk"] = nck["s_trunk"].clone(); st["s_inputs"] = nck["s_inputs"].clone()
        st["dc"] = {}
        for k, v in dc.items():
            if torch.is_tensor(v):
                st["dc"][k] = v.clone()
            elif isinstance(v, functools.partial) and "indexing_matrix" in v.keywords:   # partial(single_to_keys, indexing_matrix=..., W=, H=)
                st["dc"][k] = functools.partial(v.func, *v.args, **{**v.keywords, "indexing_matrix": v.keywords["indexing_matrix"].clone()})
            else:
                st["dc"][k] = v
        rec = _RecordingDict(feats); st["feats"] = rec
        mode_env = os.environ.get("BOLTZ_GRAPH_CAPTURE_MODE", "thread_local")
        # warm-up on a side stream (torch.cuda.graphs docs: a few eager iterations before capture, same stream as the capture)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                G.body()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        # cuBLAS/cuBLASLt workspaces are cached per (handle, stream). The warm-up allocated the capture stream's workspace from the
        # regular caching allocator; a captured GEMM bakes that pointer in, and anything that frees it later (Lightning's per-predict
        # _clear_cuda_memory = clearCublasWorkspaces + empty_cache) leaves the graph reading unmapped memory on its next replay.
        # Clearing the map HERE makes the first GEMM inside the capture allocate the workspace from the graph's private pool, whose
        # lifetime is the graph's. Scratch memory only: no numerics. (BOLTZ_GRAPH_WS_IN_POOL=0 disables this, for the falsification control.)
        if os.environ.get("BOLTZ_GRAPH_WS_IN_POOL", "1") == "1":
            _clear_cublas_workspaces_now()
        G.feats_keys = sorted(k for k in rec.accessed if k in feats and torch.is_tensor(feats[k]))
        # static feats: ALL keys kept (so any `key in feats` check sees the stock dict); tensors the network reads get static clones
        st["feats"] = {k: (feats[k].clone() if k in G.feats_keys else feats[k]) for k in feats}
        g = torch.cuda.CUDAGraph()
        mem0 = torch.cuda.memory_reserved()
        with torch.cuda.graph(g, stream=s, capture_error_mode=mode_env):
            G.body()
        torch.cuda.synchronize()
        G.graph = g; G.capture_s = round(time.time() - t0, 3); G.pool_gb = round((torch.cuda.memory_reserved() - mem0) / 1e9, 3)
        STATS["captures"] += 1; STATS["capture_s"].append(G.capture_s); STATS["capture_pool_gb"].append(G.pool_gb)
        _log(f"captured step graph in {G.capture_s}s, reserved +{G.pool_gb} GB, feats keys {G.feats_keys}")
        self._step_graph = G
        return G
    except Exception as e:
        if is_oom(e): raise                                  # out-of-memory during capture propagates: no fallback applied (a non-OOM capture failure takes the pre-draw route below, named)
        STATS["capture_error"] = traceback.format_exc()[-4000:]
        print("[boltz_graph_patch] CAPTURE FAILED -> predraw mode for the rest of the process:\n" + STATS["capture_error"], file=sys.stderr, flush=True)
        self._step_graph = None
        torch.cuda.synchronize()
        return None


def sample_graphed(self, atom_mask, num_sampling_steps=None, multiplicity=1, max_parallel_samples=None, steering_args=None,
                   **network_condition_kwargs):
    """Drop-in replacement of AtomDiffusion.sample (inference, no steering). See module docstring."""
    from boltz.model.loss.diffusionv2 import weighted_rigid_align
    from boltz.model.modules.utils import compute_random_augmentation, default

    mode = STATS["mode"]
    if (mode == "off" or self.training or steering_args is None or steering_args["fk_steering"]
            or steering_args["physical_guidance_update"]):
        STATS["stock_calls"] += 1
        return self._stock_sample(atom_mask, num_sampling_steps=num_sampling_steps, multiplicity=multiplicity,
                                  max_parallel_samples=max_parallel_samples, steering_args=steering_args, **network_condition_kwargs)
    t_s0 = time.time(); rep0, cap0 = STATS["replays"], STATS["captures"]
    from boltz.model.potentials.potentials import get_potentials
    guided = bool(steering_args["contact_guidance_update"])           # stock CLI default: True (no constraints -> exact-zero updates)
    potentials = get_potentials(steering_args, boltz2=True) if guided else None   # stock line 309 (host-side objects only)
    if max_parallel_samples is None:
        max_parallel_samples = multiplicity
    num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
    shape = (*atom_mask.shape, 3)
    device = self.device

    # ---- (1) schedule, hoisted: the same float32 values the stock loop reads with .item() (float32 -> Python float is exact) ----
    sigmas = self.sample_schedule(num_sampling_steps)
    gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
    sig = sigmas.tolist(); gam = gammas.tolist()                     # ONE host sync for the whole schedule
    step_scale = self.step_scale                                     # eval mode: never step_scale_random
    S = num_sampling_steps
    coefs = []
    for i in range(S):
        sigma_tm, sigma_t, gamma = sig[i], sig[i + 1], gam[i + 1]
        t_hat = sigma_tm * (1 + gamma)                               # stock 374 (Python double arithmetic, as stock)
        noise_var = self.noise_scale ** 2 * (t_hat ** 2 - sigma_tm ** 2)   # stock 376
        coefs.append((t_hat, sqrt(noise_var), step_scale * (sigma_t - t_hat), sigma_t))

    # ---- init noise (stock 344-345) ----
    init_sigma = sigmas[0]
    atom_coords = init_sigma * torch.randn(shape, device=device)

    # ---- (2) pre-draw every random tensor of the loop, in the stock order ----
    Rs, trs, epss = [], [], []
    for _ in range(S):
        R, tr = compute_random_augmentation(multiplicity, device=atom_coords.device, dtype=atom_coords.dtype)   # randn (m,4), randn (m,1,3)
        Rs.append(R); trs.append(tr)
        epss.append(torch.randn(shape, device=device))                                                          # randn shape
    R_all = torch.stack(Rs); tr_all = torch.stack(trs); eps_all = torch.stack(epss)
    del Rs, trs, epss

    # stock 317-325: guidance accumulator (zeros; stays exactly zero without constraints/templates, but the kernels are stock)
    scaled_guidance_update = torch.zeros((multiplicity, *atom_mask.shape[1:], 3), dtype=torch.float32, device=device) if guided else None

    sample_ids = torch.arange(multiplicity).to(device)               # stock 382 (once instead of once per step; same values)
    chunks = list(sample_ids.chunk(multiplicity % max_parallel_samples + 1))

    def guidance(atom_coords_denoised, step_idx, sigma_t, t_hat):
        """stock lines 443-473 verbatim (contact/template potentials; in-place += on the network output). Returns scaled_guidance_update."""
        steering_t = 1.0 - (step_idx / num_sampling_steps)                        # stock 375
        if guided and step_idx < num_sampling_steps - 1:
            guidance_update = torch.zeros_like(atom_coords_denoised)
            for guidance_step in range(steering_args["num_gd_steps"]):
                energy_gradient = torch.zeros_like(atom_coords_denoised)
                for potential in potentials:
                    parameters = potential.compute_parameters(steering_t)
                    if (parameters["guidance_weight"] > 0 and (guidance_step) % parameters["guidance_interval"] == 0):
                        energy_gradient += parameters["guidance_weight"] * potential.compute_gradient(
                            atom_coords_denoised + guidance_update, network_condition_kwargs["feats"], parameters)
                guidance_update -= energy_gradient
            atom_coords_denoised += guidance_update
            return guidance_update * -1 * self.step_scale * (sigma_t - t_hat) / t_hat
        return None

    def network(noisy, t_hat):
        den = torch.zeros_like(noisy)
        for ids in chunks:
            sigma = torch.full((ids.numel(),), t_hat, device=device)
            den[ids] = self._graph_precond_forward(noisy[ids], sigma, dict(multiplicity=ids.numel(), **network_condition_kwargs))
        return den

    def align(noisy, den):
        if self.alignment_reverse_diff:                              # stock 512-521
            with torch.autocast("cuda", enabled=False):
                noisy = weighted_rigid_align(noisy.float(), den.float(), atom_mask.float(), atom_mask.float())
            noisy = noisy.to(den)
        return noisy

    def euler(noisy, den, i):                                        # stock 523-528, literal expression with Python floats
        t_hat, _, coef, _ = coefs[i]
        denoised_over_sigma = (noisy - den) / t_hat
        return noisy + coef * denoised_over_sigma

    # ---- step 0, eager (stock ops; the x0_prev branch is absent in stock at step 0) ----
    t_hat, sqrt_nv, _, sigma_t = coefs[0]
    atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
    atom_coords = torch.einsum("bmd,bds->bms", atom_coords, R_all[0]) + tr_all[0]
    if guided:
        scaled_guidance_update = torch.einsum("bmd,bds->bms", scaled_guidance_update, R_all[0])   # stock 364-370
    atom_coords_noisy = atom_coords + sqrt_nv * eps_all[0]
    atom_coords_denoised = network(atom_coords_noisy, t_hat)
    sgu = guidance(atom_coords_denoised, 0, sigma_t, t_hat)
    if guided: scaled_guidance_update = sgu
    atom_coords_noisy = align(atom_coords_noisy, atom_coords_denoised)
    atom_coords = euler(atom_coords_noisy, atom_coords_denoised, 0)

    use_graph = (mode == "graph") and STATS["capture_error"] is None
    G = None
    if use_graph and S > 1:
        G = self._get_step_graph(atom_coords, atom_coords_denoised, R_all[1], tr_all[1], eps_all[1], chunks, network_condition_kwargs, scaled_guidance_update)
        use_graph = G is not None
    if use_graph:
        STATS["graph_calls"] += 1
        st = G.static
        sc_host = torch.tensor([[c[0], c[1]] for c in coefs], dtype=torch.float32)
        sc_dev = sc_host.to(device)                                   # one H2D copy: float32(t_hat), float32(sqrt(noise_var)) per step
        st["x"].copy_(atom_coords); st["den_prev"].copy_(atom_coords_denoised)
        if guided: st["sgu"].copy_(scaled_guidance_update)
        for i in range(1, S):
            st["R"].copy_(R_all[i]); st["tr"].copy_(tr_all[i]); st["eps"].copy_(eps_all[i])
            st["that"].copy_(sc_dev[i, 0]); st["sqrt_nv"].copy_(sc_dev[i, 1])
            G.graph.replay(); STATS["replays"] += 1
            sgu = guidance(st["den"], i, coefs[i][3], coefs[i][0])         # eager, stock code (op list depends on the step: weight schedule)
            if guided and sgu is not None: st["sgu"].copy_(sgu)
            noisy = align(st["noisy"], st["den"])
            atom_coords = euler(noisy, st["den"], i)
            if i + 1 < S:
                st["x"].copy_(atom_coords); st["den_prev"].copy_(st["den"])
    else:
        STATS["predraw_calls"] += 1
        for i in range(1, S):
            t_hat, sqrt_nv, _, sigma_t = coefs[i]
            R, tr = R_all[i], tr_all[i]
            atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
            atom_coords = torch.einsum("bmd,bds->bms", atom_coords, R) + tr
            atom_coords_denoised -= atom_coords_denoised.mean(dim=-2, keepdims=True)
            atom_coords_denoised = torch.einsum("bmd,bds->bms", atom_coords_denoised, R) + tr
            if guided:
                scaled_guidance_update = torch.einsum("bmd,bds->bms", scaled_guidance_update, R)
            atom_coords_noisy = atom_coords + sqrt_nv * eps_all[i]
            atom_coords_denoised = network(atom_coords_noisy, t_hat)
            sgu = guidance(atom_coords_denoised, i, sigma_t, t_hat)
            if guided and sgu is not None: scaled_guidance_update = sgu
            atom_coords_noisy = align(atom_coords_noisy, atom_coords_denoised)
            atom_coords = euler(atom_coords_noisy, atom_coords_denoised, i)
    STATS["sampler_s"].append(round(time.time() - t_s0, 4))      # host-side loop time (no sync here; the worker wrapper times with syncs)
    n_tokens = int(network_condition_kwargs["s_trunk"].shape[1])
    released = release_step_graph(self) if release_due(n_tokens) else False   # a large input's graph does not outlive its sample() (BOLTZ_GRAPH_RELEASE_MIN_TOKENS)
    STATS["last"] = {"sampler_mode": "graph" if use_graph else "predraw", "steps": S, "multiplicity": multiplicity, "n_atoms": int(shape[1]),
                     "n_tokens": n_tokens, "n_replay": STATS["replays"] - rep0, "n_capture": STATS["captures"] - cap0, "guided": guided,
                     "sampler_loop_s": STATS["sampler_s"][-1], "captures_total": STATS["captures"], "capture_s_last": (STATS["capture_s"] or [None])[-1],
                     "capture_pool_gb_last": (STATS["capture_pool_gb"] or [None])[-1], "release_min_tokens": RELEASE_MIN_TOKENS, "released": released,
                     "headroom_gated": False}                     # the hoist (outermost) adds the gate's projected / free figures for this call
    return dict(sample_atom_coords=atom_coords, diff_token_repr=None)


_APPLIED = {"done": False, "clear_ws_orig": None}
sample_graphed._bgp_patched = True


def _clear_cublas_workspaces_now():
    """Call the ORIGINAL torch._C._cuda_clearCublasWorkspaces (even while the Lightning-facing one is a no-op)."""
    f = _APPLIED["clear_ws_orig"] or getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if f is not None:
        f()


def _guard_cublas_workspaces(enable):
    """Captured cuBLAS/cuBLASLt GEMM kernels bake in the pointer of the workspace of the capture stream. pytorch_lightning's
    `_clear_cuda_memory()` (called by Strategy/Accelerator.teardown after EVERY trainer.predict) runs `torch._C._cuda_clearCublasWorkspaces()`
    + `torch.cuda.empty_cache()`, which frees that workspace -> the next replay in the same process touches unmapped/re-mapped memory
    (observed: 'CUDA error: an illegal memory access' on the first replay after a teardown). While the patch is active the clearing call is
    a no-op (the workspace stays resident: 32 MiB per (handle, stream), no numerics involved); restored by apply('off')."""
    C = torch._C
    if enable and _APPLIED["clear_ws_orig"] is None and hasattr(C, "_cuda_clearCublasWorkspaces"):
        _APPLIED["clear_ws_orig"] = C._cuda_clearCublasWorkspaces
        C._cuda_clearCublasWorkspaces = lambda: None
    elif not enable and _APPLIED["clear_ws_orig"] is not None:
        C._cuda_clearCublasWorkspaces = _APPLIED["clear_ws_orig"]; _APPLIED["clear_ws_orig"] = None


def apply(mode=None):
    """Install (mode in 'graph'|'1'|'predraw') or disable (mode in 'off'|'0'|None) the patched sampler. Idempotent; returns the mode."""
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    if mode is None:
        mode = os.environ.get("BOLTZ_GRAPH_DIFFUSION", "0")
    mode = {"1": "graph", "graph": "graph", "predraw": "predraw", "0": "off", "off": "off", "": "off"}[str(mode).lower()]
    # idempotent across re-imports / reloads: the stock method is stored ONCE on the class; a marker on the installed function says so
    if not getattr(AtomDiffusion.sample, "_bgp_patched", False):
        AtomDiffusion._stock_sample = AtomDiffusion.sample
    AtomDiffusion.sample = sample_graphed
    AtomDiffusion._graph_precond_forward = _stock_precond_forward
    AtomDiffusion._get_step_graph = _get_step_graph
    _APPLIED["done"] = True
    STATS["mode"] = mode
    _guard_cublas_workspaces(mode == "graph" and os.environ.get("BOLTZ_GRAPH_KEEP_CUBLAS_WS", "1") == "1")
    _log("mode =", mode, "| cuBLAS workspace clearing disabled:", _APPLIED["clear_ws_orig"] is not None)
    return mode


if os.environ.get("BOLTZ_GRAPH_DIFFUSION", "0") not in ("0", "off", ""):
    try:
        apply()
    except Exception as e:  # boltz not importable (e.g. sitecustomize before boltz is installed): stay inert
        print("[boltz_graph_patch] not applied:", e, file=sys.stderr)
