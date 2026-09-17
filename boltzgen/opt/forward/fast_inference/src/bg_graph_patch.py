
# bg_graph_patch.py — `graph_sampler` for BoltzGen 0.3.2 (upstream commit a3149cf): CUDA-graph capture of the design-diffusion denoiser step.
#
# What changes (a monkeypatch of AtomDiffusion.sample at inference; no upstream file is edited):
#   (1) the sigma/gamma schedule and the per-step noise variance are evaluated ONCE up front (stock: 3x .item() + math.sqrt(tensor)
#       = 4 host syncs per step). Values are the same float32 numbers the stock loop reads (float32 -> Python float is exact); the noise
#       variance is computed by the same two fp32 device ops as stock (pow(.,2) then mul), vectorised over the 500 steps, then read once.
#   (2) every random tensor of the loop is PRE-DRAWN from the CUDA generator in the stock order (init randn(shape); per step:
#       randn((m,4)) quaternion, randn((m,1,3)) translation, randn(shape) eps) -> identical CUDA RNG stream consumption and values.
#       Drawn in CHUNKS of PREDRAW_CHUNK steps as the loop reaches them (same order, nothing else draws in between -> the same
#       Philox offsets and values as drawing all 500 up front), so the pre-draw holds 32 steps of eps instead of 500 (2.7 GB at 7008 atoms x 64).
#       The eager warm-up's cached blocks are released (empty_cache) before the capture allocates the graph's private pool.
#   (3) the part of the step that is free of host syncs [centre -> random rotation/translation -> add noise -> preconditioned network
#       forward] is captured once per (shapes, strides, parameter pointers) key into a torch.cuda.CUDAGraph with static buffers and
#       replayed for every step. The rigid alignment (cuSOLVER gesvd + two torch.any() host guards) and the Euler update stay eager,
#       written as the literal stock expressions with the stock 0-d step/noise-scale tensor views and Python floats.
#   (4) the trajectory lists the sampler returns (coords_traj / x0_coords_traj: one (B*mult, M, 3) tensor per step, read only by a writer
#       asked to save trajectories) are kept on the HOST: each step's two tensors are copied off the device after the step's own host
#       synchronisation (the alignment's guards) - same values; the 2 x 500 per-step device tensors are no longer resident.
#   Nothing computed changes: same kernels, same shapes, same inputs, same step count (500), same schedules, same outputs
#   (sample_atom_coords, coords_traj, x0_coords_traj): bit-identical to the seeded stock sampler on the same GPU.
# Modes (env BG_GRAPH): "graph" (capture+replay), "predraw" (1, 2 and 4: eager network), "off" (stock). Falls back to predraw on capture error.
import os, sys, time, functools, traceback
from math import sqrt
import torch
from opt_core.oom import is_oom

PREDRAW_CHUNK = 32                                     # steps of loop randomness drawn per chunk (stock order within and across chunks)
TRAJ_DEVICE = "cpu"                                    # where the per-step trajectory tensors the sampler returns are kept (their values are the device tensors', moved)
STATS = {"mode": "off", "captures": 0, "replays": 0, "graph_calls": 0, "predraw_calls": 0, "stock_calls": 0,
         "capture_error": None, "capture_s": [], "capture_pool_gb": [], "sampler_s": [], "last": None, "cache_hits": 0,
         "predraw_chunk": PREDRAW_CHUNK, "predraw_chunks": 0, "traj_device": TRAJ_DEVICE}
_MAX_GRAPHS = 16                                       # graphs kept per process (least recently used evicted)
_STEPS = {"design"}                                    # pipeline steps the patch is active in


def _log(msg):
    print("[bg_graph_patch] " + msg, file=sys.stderr, flush=True)


def _sig(t):
    return (tuple(t.shape), str(t.dtype), t.device.index, tuple(t.stride()))


class _RecordingDict(dict):
    """feats wrapper that records which keys the network reads (warm-up only)."""
    def __init__(self, d):
        super().__init__(d); self.accessed = set()
    def __getitem__(self, k):
        self.accessed.add(k); return super().__getitem__(k)
    def get(self, k, default=None):
        self.accessed.add(k); return super().get(k, default)
    def __contains__(self, k):
        self.accessed.add(k); return super().__contains__(k)


class _StepGraph:
    """Static buffers + captured CUDA graph of one denoiser step for a fixed key."""
    def __init__(self, diff, key):
        self.diff = diff; self.key = key; self.graph = None; self.static = {}; self.feats_keys = None
        self.capture_s = None; self.pool_gb = None; self.last_used = 0

    def cond_kwargs(self):
        S = self.static
        return dict(s_trunk=S["s_trunk"], s_inputs=S["s_inputs"], feats=S["feats"], diffusion_conditioning=S["dc"])

    def body(self):
        """Stock AtomDiffusion.sample loop lines (a3149cf diffusion.py 572-599) from center() to the network call, for one step,
        with R/tr/eps/t_hat/sqrt(noise_var)/noise_scale read from static buffers."""
        from boltzgen.model.modules.utils import center
        S = self.static; diff = self.diff
        atom_coords = center(S["x"], S["atom_mask"])                                        # stock 577
        if diff.coordinate_augmentation_inference:                                          # stock 579-585
            atom_coords = torch.einsum("bmd,bds->bms", atom_coords, S["R"]) + S["tr"]
        eps = S["ns"] * S["sqrt_nv"]                                                        # stock 587: noise_scale(0-d) * sqrt(noise_var)
        eps = eps * S["eps"]                                                                #           * randn(shape)
        atom_coords_noisy = atom_coords + eps                                               # stock 588
        sigma = S["sigma"]; sigma.copy_(S["that"].expand(sigma.shape[0]))                   # == torch.full((batch,), t_hat) in stock 390
        with torch.no_grad():                                                               # stock 590-599
            den, _net_out = diff.preconditioned_network_forward(
                atom_coords_noisy, sigma, training=False,
                network_condition_kwargs=dict(multiplicity=S["multiplicity"], **self.cond_kwargs()))
        S["noisy"].copy_(atom_coords_noisy); S["den"].copy_(den)


def _clear_cublas_workspaces_now():
    f = _APPLIED["clear_ws_orig"] or getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if f is not None:
        f()


def _get_step_graph(self, x, R, tr, eps, atom_mask, multiplicity, nck):
    dc = nck["diffusion_conditioning"]; feats = nck["feats"]
    param_key = hash(tuple(p.data_ptr() for p in self.score_model.parameters()))
    key = (_sig(x), _sig(R), _sig(tr), _sig(eps), _sig(atom_mask), int(multiplicity), _sig(nck["s_trunk"]), _sig(nck["s_inputs"]),
           tuple((k, _sig(v)) for k, v in sorted(dc.items()) if torch.is_tensor(v)), param_key,
           torch.is_autocast_enabled(), torch.get_float32_matmul_precision())
    cache = getattr(self, "_bg_graphs", None)
    if cache is None:
        cache = self._bg_graphs = {}
    G = cache.get(key)
    if G is not None and any((k not in feats) or (tuple(feats[k].shape) != tuple(G.static["feats"][k].shape)) for k in G.feats_keys):
        _log("cache key hit but feats shapes differ -> recapture"); del cache[key]; torch.cuda.synchronize(); G = None
    if G is not None:
        st = G.static                                       # same shapes: refresh the per-design conditioning values only
        st["s_trunk"].copy_(nck["s_trunk"]); st["s_inputs"].copy_(nck["s_inputs"]); st["atom_mask"].copy_(atom_mask)
        for k, v in dc.items():
            if torch.is_tensor(v):
                st["dc"][k].copy_(v)
            elif isinstance(v, functools.partial) and "indexing_matrix" in v.keywords:
                st["dc"][k].keywords["indexing_matrix"].copy_(v.keywords["indexing_matrix"])
        for k in feats:
            if k in G.feats_keys:
                st["feats"][k].copy_(feats[k])
            else:
                st["feats"][k] = feats[k]
        STATS["cache_hits"] += 1; G.last_used = time.time()
        return G
    if len(cache) >= _MAX_GRAPHS:                           # evict the least recently used graph
        old_key = min(cache, key=lambda k: cache[k].last_used)
        del cache[old_key]; torch.cuda.synchronize(); _clear_cublas_workspaces_now()
    G = _StepGraph(self, key)
    try:
        t0 = time.time(); st = G.static
        st["x"] = x.clone(); st["R"] = R.clone(); st["tr"] = tr.clone(); st["eps"] = eps.clone(); st["atom_mask"] = atom_mask.clone()
        st["noisy"] = torch.empty_like(x); st["den"] = torch.empty_like(x); st["multiplicity"] = int(multiplicity)
        st["that"] = torch.zeros((), dtype=torch.float32, device=x.device); st["sqrt_nv"] = torch.zeros((), dtype=torch.float32, device=x.device)
        st["ns"] = torch.zeros((), dtype=torch.float32, device=x.device)
        st["sigma"] = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
        st["s_trunk"] = nck["s_trunk"].clone(); st["s_inputs"] = nck["s_inputs"].clone()
        st["dc"] = {}
        for k, v in dc.items():
            if torch.is_tensor(v):
                st["dc"][k] = v.clone()
            elif isinstance(v, functools.partial) and "indexing_matrix" in v.keywords:
                st["dc"][k] = functools.partial(v.func, *v.args, **{**v.keywords, "indexing_matrix": v.keywords["indexing_matrix"].clone()})
            else:
                st["dc"][k] = v
        rec = _RecordingDict(feats); st["feats"] = rec
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        n_warm = 3 if STATS["captures"] == 0 else 1
        with torch.cuda.stream(s):
            for _ in range(n_warm):
                G.body()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        _clear_cublas_workspaces_now()                      # workspace of the first captured GEMM is allocated from the graph pool
        torch.cuda.empty_cache()                            # the warm-up's (and the trunk's) freed blocks leave the default pool's cache before the capture fills the
                                                            # graph's private pool: one step working set stays reserved, not two (numerics untouched; once per capture)
        G.feats_keys = sorted(k for k in rec.accessed if k in feats and torch.is_tensor(feats[k]))
        st["feats"] = {k: (feats[k].clone() if k in G.feats_keys else feats[k]) for k in feats}
        g = torch.cuda.CUDAGraph(); mem0 = torch.cuda.memory_reserved()
        with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"):   # private pool per graph
            G.body()
        torch.cuda.synchronize()
        G.graph = g; G.capture_s = round(time.time() - t0, 3); G.pool_gb = round((torch.cuda.memory_reserved() - mem0) / 1e9, 3)
        G.last_used = time.time()
        STATS["captures"] += 1; STATS["capture_s"].append(G.capture_s); STATS["capture_pool_gb"].append(G.pool_gb)
        _log(f"captured step graph #{STATS['captures']} in {G.capture_s}s (+{G.pool_gb} GB reserved; x {tuple(x.shape)}; feats {G.feats_keys})")
        cache[key] = G
        return G
    except Exception as e:
        if is_oom(e): raise
        STATS["capture_error"] = traceback.format_exc()[-4000:]
        print("[bg_graph_patch] CAPTURE FAILED -> predraw mode for the rest of the process:\n" + STATS["capture_error"], file=sys.stderr, flush=True)
        torch.cuda.synchronize()
        return None


def sample_patched(self, atom_mask, num_sampling_steps=None, multiplicity=1, step_scale=None, noise_scale=None,
                   inference_logging=False, **network_condition_kwargs):
    """Drop-in replacement of AtomDiffusion.sample (boltzgen a3149cf diffusion.py 501-631) for inference."""
    from boltzgen.model.modules.utils import center, compute_random_augmentation, default
    from boltzgen.model.loss.diffusion import weighted_rigid_align
    import numpy as np
    mode = STATS["mode"]
    if mode == "off" or self.training or os.environ.get("BOLTZGEN_PIPELINE_STEP", "design") not in _STEPS:
        STATS["stock_calls"] += 1
        return self._bg_stock_sample(atom_mask, num_sampling_steps=num_sampling_steps, multiplicity=multiplicity, step_scale=step_scale,
                                     noise_scale=noise_scale, inference_logging=inference_logging, **network_condition_kwargs)
    t_s0 = time.time(); rep0, cap0 = STATS["replays"], STATS["captures"]
    # ---- stock 511-530: step/noise scale vectors (same tensors) ----
    if self.training and self.step_scale_random is not None:
        step_scales = np.random.choice(self.step_scale_random) * torch.ones(num_sampling_steps, device=self.device, dtype=torch.float32)
    elif self.step_scale_function == "beta":
        step_scales = self.beta_step_scale_schedule(num_sampling_steps)
    else:
        step_scales = default(step_scale, self.step_scale) * torch.ones(num_sampling_steps, device=self.device, dtype=torch.float32)
    if self.noise_scale_function == "constant":
        noise_scales = default(noise_scale, self.noise_scale) * torch.ones(num_sampling_steps, device=self.device, dtype=torch.float32)
    elif self.noise_scale_function == "beta":
        noise_scales = self.beta_noise_scale_schedule(num_sampling_steps)
    else:
        raise ValueError(f"Invalid noise scale schedule: {self.noise_scale_function}")
    num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
    shape = (*atom_mask.shape, 3)
    if self.sampling_schedule == "af3":
        sigmas = self.sample_schedule_af3(num_sampling_steps)
    elif self.sampling_schedule == "dilated":
        sigmas = self.sample_schedule_dilated(num_sampling_steps)
    gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
    S = num_sampling_steps
    # ---- (1) schedule hoisted: ONE host read of the same float32 values ----
    sig = sigmas.tolist(); gam = gammas.tolist()
    t_hats = [sig[i] * (1 + gam[i + 1]) for i in range(S)]                                   # stock 574 (Python double arithmetic)
    d_py = [t_hats[i] ** 2 - sig[i] ** 2 for i in range(S)]                                  # stock 575 inner term (Python double)
    # stock 575: noise_var = noise_scale**2 * (t_hat**2 - sigma_tm**2)  [0-d fp32 tensor ops: pow(.,2) then mul by the double scalar]
    noise_var_all = noise_scales ** 2 * torch.tensor(d_py, dtype=torch.float32, device=self.device)
    sqrt_nvs = [sqrt(v) for v in noise_var_all.tolist()]                                     # stock 587: math.sqrt(noise_var) (double)
    # ---- init noise (stock 553-554) ----
    init_sigma = sigmas[0]
    atom_coords = init_sigma * torch.randn(shape, device=self.device)
    feats = network_condition_kwargs["feats"]
    # ---- (2) pre-draw the loop's randomness in the stock order, PREDRAW_CHUNK steps at a time (draw_chunk(i0) is called when the loop
    #      reaches step i0; nothing else consumes the CUDA generator in between, so the stream is consumed exactly as by the stock loop) ----
    aug_dtype, aug_device = atom_coords.dtype, atom_coords.device
    def draw_chunk(i0):
        k = min(PREDRAW_CHUNK, S - i0)
        Rs, trs, epss = [], [], []
        for _ in range(k):
            if self.coordinate_augmentation_inference:
                R, tr = compute_random_augmentation(multiplicity, device=aug_device, dtype=aug_dtype)
                Rs.append(R); trs.append(tr)
            epss.append(torch.randn(shape, device=self.device))
        eps_c = torch.stack(epss); del epss
        if self.coordinate_augmentation_inference:
            R_c = torch.stack(Rs); tr_c = torch.stack(trs); del Rs, trs
        else:
            R_c = torch.eye(3, device=self.device, dtype=aug_dtype)[None].repeat(multiplicity, 1, 1)[None].repeat(k, 1, 1, 1)
            tr_c = torch.zeros((k, multiplicity, 1, 3), device=self.device, dtype=aug_dtype)
        STATS["predraw_chunks"] += 1
        return R_c, tr_c, eps_c
    torch.cuda.empty_cache()                              # the trunk's freed activations leave the allocator's cache before the sampler's working set is allocated (once per call)
    R_c, tr_c, eps_c = draw_chunk(0)
    nck = network_condition_kwargs
    keep = lambda t: t.to(TRAJ_DEVICE)                    # (4) a trajectory entry: the step's tensor, moved off the device (values unchanged)
    coords_traj = [keep(atom_coords)]; x0_coords_traj = []

    def align_and_euler(noisy, den, i):
        """stock 601-615 verbatim: rigid alignment (eager; host guards + cuSOLVER) and the Euler update with the stock 0-d tensor views."""
        t_hat = t_hats[i]; sigma_t = sig[i + 1]; step_scale_i = step_scales[i]
        if self.alignment_reverse_diff:
            with torch.autocast("cuda", enabled=False):
                noisy = weighted_rigid_align(noisy.float(), den.float(), atom_mask.float(), atom_mask.float())
            noisy = noisy.to(den)
        denoised_over_sigma = (noisy - den) / t_hat
        return noisy + step_scale_i * (sigma_t - t_hat) * denoised_over_sigma

    use_graph = (mode == "graph") and STATS["capture_error"] is None
    G = None
    if use_graph:
        G = _get_step_graph(self, atom_coords, R_c[0], tr_c[0], eps_c[0], atom_mask, multiplicity, nck)
        use_graph = G is not None
    if use_graph:
        STATS["graph_calls"] += 1
        st = G.static
        sc = torch.tensor([[t_hats[i], sqrt_nvs[i]] for i in range(S)], dtype=torch.float32, device=self.device)   # one H2D copy
        for i in range(S):
            j = i % PREDRAW_CHUNK
            if j == 0 and i:
                R_c, tr_c, eps_c = draw_chunk(i)
            st["x"].copy_(atom_coords); st["R"].copy_(R_c[j]); st["tr"].copy_(tr_c[j]); st["eps"].copy_(eps_c[j])
            st["that"].copy_(sc[i, 0]); st["sqrt_nv"].copy_(sc[i, 1]); st["ns"].copy_(noise_scales[i])
            G.graph.replay(); STATS["replays"] += 1
            den = st["den"].clone()
            atom_coords = align_and_euler(st["noisy"].clone(), den, i)
            coords_traj.append(keep(atom_coords)); x0_coords_traj.append(keep(den))
    else:
        STATS["predraw_calls"] += 1
        for i in range(S):
            j = i % PREDRAW_CHUNK
            if j == 0 and i:
                R_c, tr_c, eps_c = draw_chunk(i)
            t_hat = t_hats[i]
            atom_coords = center(atom_coords, atom_mask)
            if self.coordinate_augmentation_inference:
                atom_coords = torch.einsum("bmd,bds->bms", atom_coords, R_c[j]) + tr_c[j]
            eps = noise_scales[i] * sqrt_nvs[i] * eps_c[j]
            atom_coords_noisy = atom_coords + eps
            with torch.no_grad():
                atom_coords_denoised, _net_out = self.preconditioned_network_forward(
                    atom_coords_noisy, t_hat, training=False, network_condition_kwargs=dict(multiplicity=multiplicity, **nck))
            atom_coords = align_and_euler(atom_coords_noisy, atom_coords_denoised, i)
            coords_traj.append(keep(atom_coords)); x0_coords_traj.append(keep(atom_coords_denoised))
    coords_traj.append(keep(atom_coords))
    STATS["sampler_s"].append(round(time.time() - t_s0, 4))
    STATS["last"] = {"sampler_mode": "graph" if use_graph else "predraw", "steps": S, "multiplicity": multiplicity, "n_atoms": int(shape[1]),
                     "n_replay": STATS["replays"] - rep0, "n_capture": STATS["captures"] - cap0, "sampler_loop_s": STATS["sampler_s"][-1],
                     "captures_total": STATS["captures"], "cache_hits": STATS["cache_hits"], "predraw_chunk": PREDRAW_CHUNK, "traj_device": TRAJ_DEVICE}
    return dict(sample_atom_coords=atom_coords, coords_traj=coords_traj, x0_coords_traj=x0_coords_traj)


sample_patched._bg_patched = True
_APPLIED = {"done": False, "clear_ws_orig": None, "pool": None}


def _guard_cublas_workspaces(enable):
    """pytorch_lightning's _clear_cuda_memory (after every trainer.predict) frees cuBLAS workspaces that captured GEMMs point into;
    while graphs are live that call is made a no-op (scratch memory only, no numerics)."""
    C = torch._C
    if enable and _APPLIED["clear_ws_orig"] is None and hasattr(C, "_cuda_clearCublasWorkspaces"):
        _APPLIED["clear_ws_orig"] = C._cuda_clearCublasWorkspaces
        C._cuda_clearCublasWorkspaces = lambda: None
    elif not enable and _APPLIED["clear_ws_orig"] is not None:
        C._cuda_clearCublasWorkspaces = _APPLIED["clear_ws_orig"]; _APPLIED["clear_ws_orig"] = None


def release(model=None):
    """Free all captured graphs (e.g. after the design step of an in-process pipeline) and restore torch's workspace clearing."""
    n = 0
    from boltzgen.model.modules.diffusion import AtomDiffusion
    objs = [model.structure_module] if model is not None and hasattr(model, "structure_module") else []
    for sm in objs:
        cache = getattr(sm, "_bg_graphs", None)
        if cache:
            n += len(cache); cache.clear()
    torch.cuda.synchronize()
    _guard_cublas_workspaces(False)
    _clear_cublas_workspaces_now(); torch.cuda.empty_cache()
    _log(f"released {n} graph(s)")
    return n


def apply(mode=None):
    from boltzgen.model.modules.diffusion import AtomDiffusion
    if mode is None:
        mode = os.environ.get("BG_GRAPH", "off")
    mode = {"1": "graph", "graph": "graph", "predraw": "predraw", "0": "off", "off": "off"}.get(str(mode).lower(), "off")
    STATS["mode"] = mode
    if mode != "off" and not getattr(AtomDiffusion.sample, "_bg_patched", False):
        AtomDiffusion._bg_stock_sample = AtomDiffusion.sample          # the genuine stock function (never a patched one)
        AtomDiffusion.sample = sample_patched
        _APPLIED["done"] = True
    elif mode != "off" and not hasattr(AtomDiffusion, "_bg_stock_sample"):
        raise RuntimeError("bg_graph_patch: AtomDiffusion.sample is patched but no stock alias exists")
    if mode == "graph":
        _guard_cublas_workspaces(True)
    _log(f"mode={mode}")
    return mode
