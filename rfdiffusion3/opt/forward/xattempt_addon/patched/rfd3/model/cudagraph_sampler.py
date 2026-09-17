"""CUDA-graph rollout for the RFD3 default inference sampler.

Enable with RFD3_CUDAGRAPH=1 (default 0 = stock code path, nothing in this module is used).

What it does
------------
`SampleDiffusionWithMotifGraph` re-implements `SampleDiffusionWithMotif.sample_diffusion_like_af3`
with the per-step denoiser call `diffusion_module(X_noisy_L, t, f, **initializer_outputs)` replaced
by the replay of ONE captured CUDA graph per (batch size D, structural signature of the design).
Everything the stock loop does OUTSIDE that call is taken from upstream at upstream's own points:
the initial structure (`_get_initial_structure`), the origin jitter (`s_jitter_origin`), the
per-step re-centring + random rotation / translation around the motif (`allow_realignment`, with
the motif-fix schedule `fraction_of_steps_to_fix_motif` feeding its two arguments) and the closing
motif re-insertion + alignment, and the partial-diffusion schedule (`f["partial_t"]` through
`_construct_inference_noise_schedule`). Two roll-out kinds are NOT driven here and go to the stock
sampler eagerly, counted and named (`declined_options`: classifier-free guidance — a second denoiser
pass per step on stripped features, which one captured step per roll-out has no slot for; the
chunked pair embedder of upstream's low-memory mode — `P_LL=None`, the pair track recomputed inside
every step); the kit's modes refuse both before a run starts, so under a mode they are not reached.
The arithmetic of the sampler is unchanged:

* all Gaussian draws are taken from the default generators in the stock order and at the stock
  points (initial noise, origin jitter, then per step the realignment's rotation and translation
  when it is on, then the (D, L, 3) step noise); the denoiser — eager, warm-up, capture or replay —
  consumes no RNG at inference, so the generator streams are identical to stock;
* the boolean-mask index_put `x[..., is_motif_atom_with_fixed_coord, :] = 0` is done with the
  integer indices of the same mask (same elements, same values);
* `gamma = gamma_0 if c_t > gamma_min else 0` is evaluated on a host copy of the (float32) noise
  schedule instead of a device tensor (same comparison, no per-step sync);
* the per-step `torch.softmax(sequence_logits).cpu()` read-back (only consumed by the entropy
  trajectory, which nothing downstream uses at inference) is moved OUT of the step loop: the
  softmax is still computed on the GPU at every step and the entropy is computed on the CPU with
  the stock expression at the end of the rollout (identical values, one device->host copy);
* the duplicate-attention-index assertion inside `get_sparse_attention_indices` is evaluated on
  the device into an accumulator during the captured step and checked once after the rollout.

Inside the captured region a handful of host synchronisations (`.item()`, `int(tensor)`,
`torch.unique`, boolean-mask indexing -> `nonzero`, `torch.tensor(scalar, device=cuda)`) are
replaced by values memoised during the eager warm-up iterations (see `cg_*` helpers used by the
flag-guarded branches in blocks.py / block_utils.py / attention.py / encoders.py /
pairformer_layers.py / RFD3_diffusion_module.py). All of them are functions of the static
(per-design) structure only; the memoised values are exactly what the stock code computes.

Graph reuse across designs: a graph is keyed by D, the shapes/dtypes of every tensor in `f` and
`initializer_outputs`, the sha256 of every non-floating-point tensor in `f` (atom/token maps and
masks), `n_recycle` and the attention env flags. On a key hit the new design's tensors are copied
into the static buffers and the graph is replayed; on a miss the previous graph is freed and a new
one is captured (2 eager warm-up iterations + 1 capture).
"""
import gc
import hashlib
import logging
import os
import time

import torch

from rfd3.model.hoist import nodynamo  # [RFD3_COMPILE] the memo helpers below are dynamo boundaries (identity unless RFD3_COMPILE=1)
from opt_core.oom import is_oom  # the core's one out-of-memory classifier: the capture fallback re-raises an out-of-memory before rerouting anything

logger = logging.getLogger(__name__)

_STATE = {"mode": None, "phase": None, "entry": None}  # mode: None | "graph" (the model code branches on it: one value through warm-up AND capture, so a compiled
# step sees the same guards in both); phase: None | "record" (eager warm-up, memoise) | "capture" (graph capture) — read only by the helpers below
GRAPH_STATS = {"installs": 0, "captures": 0, "hits": 0, "fallbacks": 0, "declined": 0, "declined_by": "none"}  # process tally (every wrapper adds to it): the evidence a run record
# carries. fallbacks = captures that FAILED and were served by the eager denoiser; declined / declined_by = roll-outs this sampler does not capture BY DESIGN (the options
# `declined_options` names), served by the stock sampler eagerly — the eager path, numerically what the stock roll-out computes


def enabled() -> bool:
    return os.environ.get("RFD3_CUDAGRAPH", "0") == "1"


def graph_mode():
    """None (stock code everywhere) | 'graph' (inside a graph roll-out's warm-up or capture: the flag-guarded branches are taken). The
    record/capture distinction is graph_phase()."""
    return _STATE["mode"]


def graph_phase():
    """None | 'record' (eager warm-up, memoise) | 'capture' (graph capture)."""
    return _STATE["phase"]


def _set_phase(phase):
    _STATE["phase"] = phase
    _STATE["mode"] = None if phase is None else "graph"


class _Entry:
    def __init__(self):
        self.memo = {}
        self.graph = None
        self.dup_acc = None
        self.static_f = None
        self.static_io = None
        self.x_in = self.t_in = self.x_out = self.logits_out = self.idx_out = None
        self.key = None


def _entry():
    e = _STATE["entry"]
    assert e is not None, "cudagraph helpers used outside a graph rollout"
    return e


def _memo(key, compute, src=None):
    e = _entry()
    mode = _STATE["phase"]
    if key in e.memo:
        val, src_ref = e.memo[key]
        if mode == "record" and src_ref is not None and src is not None:
            # eager sanity check: a memo hit must refer to identical source data (guards ptr aliasing)
            assert src_ref.shape == src.shape and torch.equal(src_ref, src), f"memo aliasing detected for {key}"
        return val
    assert mode == "record", f"memo miss in mode={mode} for key={key} (capture must not compute new host values)"
    val = compute()
    e.memo[key] = (val, src.detach().clone() if (src is not None and src.numel() <= 1 << 22) else None)
    return val


class Unsupported(RuntimeError):
    """Raised (in record mode) by a code path that cannot be captured; the rollout falls back to the eager denoiser."""


def _tkey(name, t):
    # keyed by name/shape/dtype (NOT data_ptr: several call sites rebuild e.g. torch.arange(I) every call);
    # the record-mode src equality check in _memo guards against two different tensors sharing a key.
    return (name, tuple(t.shape), str(t.dtype))


# ---------------- helpers used by the flag-guarded branches in the model code ----------------

@nodynamo
def cg_int_max_plus_1(t):
    """== int(t.max().item()) + 1 (stock), memoised per tensor identity during warm-up."""
    return _memo(_tkey("imax1", t), lambda: int(t.max().item()) + 1, src=t)


@nodynamo
def cg_nonzero_1d(mask, name="nz"):
    """== mask.nonzero().squeeze(-1) for a 1-D bool mask, memoised."""
    return _memo(_tkey(name, mask), lambda: mask.nonzero(as_tuple=False).squeeze(-1).contiguous(), src=mask)


@nodynamo
def cg_const(value, dtype, device, shape=()):
    """Device scalar/1-element constant, created once (avoids an H2D copy inside the captured region)."""
    key = ("const", float(value), str(dtype), str(device), tuple(shape))
    return _memo(key, lambda: torch.full(shape, float(value), dtype=dtype, device=device) if shape else torch.tensor(value, dtype=dtype, device=device))


@nodynamo
def cg_valid_mask(tok_idx, n_atoms_per_tok_max, compute):
    return _memo(_tkey(("vmask", n_atoms_per_tok_max), tok_idx), compute, src=tok_idx)


@nodynamo
def cg_flat_idx(valid_mask):
    return _memo(_tkey("flat", valid_mask), lambda: valid_mask.flatten().nonzero(as_tuple=False).squeeze(1).contiguous(), src=valid_mask)


@nodynamo
def cg_bool(key, compute):
    return _memo(key, compute)


@nodynamo
def cg_dense_decision(D, L, k, H, full, compute):
    return _memo(("dense", D, L, k, H, bool(full)), compute)


@nodynamo
def cg_dup_accumulate(flag):
    """Deferred assertion: OR the per-call duplicate flag (0-d bool tensor) into the entry accumulator."""
    e = _entry()
    e.dup_acc.logical_or_(flag)


@nodynamo
def cg_index_select_mask(x, mask, dim=-2, name="isel"):
    """== x[..., mask, :] for a 1-D bool mask along `dim` (index_select on the memoised nonzero indices)."""
    return x.index_select(dim, cg_nonzero_1d(mask, name=name))


# ---------------------------------------- the sampler ----------------------------------------

def _signature(f, io, D, n_recycle):
    h = hashlib.sha256()
    h.update(f"D={D};n_recycle={n_recycle};dense={os.environ.get('RFD3_DENSE_SDPA_ATTENTION', '1')};lowmem={os.environ.get('RFD3_LOW_MEMORY_MODE', '0')};".encode())
    from rfd3.model import hoist as _hoist  # [RFD3_HOIST] hoist configuration is part of the captured graph

    h.update(_hoist.signature_token().encode())
    h.update(f"compile={int(bool(_hoist.COMPILE))};token_sdpa={int(bool(_hoist.TOKEN_SDPA))};".encode())  # [RFD3_COMPILE][RFD3_TOKEN_SDPA] part of what the graph captured
    for dname, d in (("f", f), ("io", io)):
        for k in sorted(d.keys()):
            v = d[k]
            if torch.is_tensor(v):
                h.update(f"{dname}.{k}:{tuple(v.shape)}:{v.dtype};".encode())
                if not v.is_floating_point():
                    h.update(v.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(f"{dname}.{k}:{type(v).__name__}:{v!r};".encode())
    return h.hexdigest()


def declined_options(s, f, initializer_outputs, ref_initializer_outputs, coords):
    """The roll-out kinds this sampler does not drive, by name ([] = driven): classifier-free guidance (upstream's second, unconditioned denoiser
    pass per step on the stripped features — `ref_initializer_outputs` present or the sampler's own switch on: one captured step per roll-out has
    no slot for it), the chunked pair embedder of upstream's low-memory mode (`P_LL=None`, the pair track recomputed in row blocks inside every
    step), a tensor not on CUDA. The kit's modes refuse the first two before a run starts (rfdiffusion3_opt.modes: the design line's
    `inference_sampler.use_classifier_free_guidance` / `low_memory_mode`, the engine's own decision at initialize); reached anyway (upstream's API
    driven by hand), the stock sampler runs that roll-out eagerly and the tally names it (GRAPH_STATS declined / declined_by). Realignment, origin
    jitter, partial diffusion and the motif-fix schedule are driven: they live outside the denoiser call (sample_diffusion_like_af3 below)."""
    checks = (("use_classifier_free_guidance", bool(s.use_classifier_free_guidance) or ref_initializer_outputs is not None),
              ("chunked_pairwise_embedder", "chunked_pairwise_embedder" in initializer_outputs),
              ("cpu", not coords.is_cuda))
    return [name for name, hit in checks if hit]


class SampleDiffusionWithMotifGraph:
    """Mixin-style wrapper: built by `install()` on an existing SampleDiffusionWithMotif instance."""

    def __init__(self, stock_sampler):
        self.stock = stock_sampler
        self.entry = None
        self.stats = {"captures": 0, "hits": 0, "capture_s": [], "fallbacks": 0, "declined": 0}

    def __getattr__(self, k):
        # only attributes the wrapper does not hold reach here; `stock` itself is absent on an instance made without __init__
        # (copy.deepcopy / pickle reconstruct the object and probe `__setstate__` / `__deepcopy__` first — the engine deep-copies
        # the model for its EMA weights, foundry/training/EMA.py): a plain AttributeError there, never a recursion into this method
        stock = self.__dict__.get("stock")
        if stock is None:
            raise AttributeError(k)
        return getattr(stock, k)

    # --- graph management ---
    def _capture(self, f, diffusion_module, io, D, L, x_probe, t_probe):
        s = self.stock
        if self.entry is not None:
            self.entry = None
            gc.collect()
            torch.cuda.synchronize()
        e = _Entry()
        dev = x_probe.device
        e.static_f = {k: (v.detach().clone().contiguous() if torch.is_tensor(v) else v) for k, v in f.items()}
        e.static_io = {k: (v.detach().clone().contiguous() if torch.is_tensor(v) else v) for k, v in io.items()}
        e.x_in = x_probe.detach().clone().contiguous()
        e.t_in = t_probe.detach().clone().contiguous()
        e.dup_acc = torch.zeros((), dtype=torch.bool, device=dev)
        ac_on = torch.is_autocast_enabled("cuda") if hasattr(torch, "is_autocast_enabled") else False
        try:
            ac_dtype = torch.get_autocast_dtype("cuda")
        except Exception:
            ac_dtype = torch.bfloat16
        e.autocast = (bool(ac_on), str(ac_dtype))
        e.autocast_on = bool(ac_on)  # [RFD3_HOIST] used by hoist.hoist_refresh on entry reuse
        e.autocast_dtype = ac_dtype

        def fwd():
            with torch.autocast("cuda", enabled=bool(ac_on), dtype=ac_dtype, cache_enabled=False):
                return diffusion_module(X_noisy_L=e.x_in, t=e.t_in, f=e.static_f, n_recycle=s.n_recycle, **e.static_io)

        _STATE["entry"] = e
        t0 = time.time()
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            _set_phase("record")
            with torch.cuda.stream(side):
                for _ in range(2):
                    out = fwd()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            _set_phase("capture")
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = fwd()
            torch.cuda.synchronize()
        finally:
            _set_phase(None)
        e.graph = g
        e.x_out = out["X_L"]
        e.logits_out = out["sequence_logits_I"]
        e.idx_out = out["sequence_indices_I"]
        e.n_recycle = s.n_recycle
        self.entry = e
        self.stats["captures"] += 1
        GRAPH_STATS["captures"] += 1
        self.stats["capture_s"].append(round(time.time() - t0, 3))
        logger.info(f"[RFD3_CUDAGRAPH] captured denoiser graph D={D} L={L} in {time.time() - t0:.2f}s (autocast={e.autocast})")
        return e

    def _get_entry(self, f, diffusion_module, io, D, L, x_probe, t_probe):
        key = _signature(f, io, D, self.stock.n_recycle)
        e = self.entry
        if e is not None and e.key == key:
            for k, v in f.items():
                if torch.is_tensor(v):
                    e.static_f[k].copy_(v)
            for k, v in io.items():
                if torch.is_tensor(v):
                    e.static_io[k].copy_(v)
            e.dup_acc.zero_()
            # [RFD3_HOIST] the captured graph reads hoisted tensors from entry-owned buffers: recompute them for this roll-out's inputs
            from rfd3.model import hoist as _hoist

            e.hoist_refreshed = _hoist.hoist_refresh(e)
            self.stats["hits"] += 1
            GRAPH_STATS["hits"] += 1
            return e
        e = self._capture(f, diffusion_module, io, D, L, x_probe, t_probe)
        e.key = key
        return e

    # --- rollout ---
    def sample_diffusion_like_af3(self, *, f, diffusion_module, diffusion_batch_size, coord_atom_lvl_to_be_noised,
                                  initializer_outputs, ref_initializer_outputs, f_ref, **_):
        s = self.stock
        declined = declined_options(s, f, initializer_outputs, ref_initializer_outputs, coord_atom_lvl_to_be_noised)
        if declined:  # not driven here (declined_options): the stock sampler runs this roll-out eagerly; counted and named (GRAPH_STATS declined / declined_by), never silent
            self.stats["declined"] += 1
            GRAPH_STATS["declined"] += 1
            GRAPH_STATS["declined_by"] = ",".join(sorted(set([w for w in GRAPH_STATS["declined_by"].split(",") if w and w != "none"] + declined)))
            logger.warning(f"[RFD3_CUDAGRAPH] declined by design ({','.join(declined)}): the stock sampler runs this roll-out eagerly")
            return s.sample_diffusion_like_af3(f=f, diffusion_module=diffusion_module, diffusion_batch_size=diffusion_batch_size,
                                               coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised, initializer_outputs=initializer_outputs,
                                               ref_initializer_outputs=ref_initializer_outputs, f_ref=f_ref)
        use_graph = True  # False after a capture failure: the eager denoiser serves the rest of this roll-out (same arithmetic, same noise)
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]
        dev = coord_atom_lvl_to_be_noised.device
        noise_schedule = s._construct_inference_noise_schedule(device=dev, partial_t=f.get("partial_t", None))  # partial diffusion: upstream's own schedule filter (inference_sampler.py:165-168)
        ns_host = noise_schedule.detach().cpu().tolist()  # host copy for the python-level gamma decision (one sync)
        L = f["ref_element"].shape[0]
        D = diffusion_batch_size
        T = len(noise_schedule)
        fixed_idx = is_motif_atom_with_fixed_coord.nonzero(as_tuple=False).squeeze(-1)
        realign = bool(s.allow_realignment)
        threshold_step = (T - 1) * s.fraction_of_steps_to_fix_motif  # the motif-fix schedule: read only by the realignment call's two arguments (inference_sampler.py:194, 211, 214)
        if realign:  # upstream's own realignment statements (the module that built this wrapper — already imported; named at call time, no import cycle)
            from rfd3.model.inference_sampler import centre_random_augment_around_motif, weighted_rigid_align

        # --- Gaussian draws in the stock order, each at upstream's own point: the initial noise, the origin jitter, then per step the
        # realignment's rotation + translation (when on) and the (D, L, 3) step noise. The denoiser consumes no RNG. ---
        noise = noise_schedule[0] * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=dev)
        noise[:, fixed_idx, :] = 0
        X_L = noise + coord_atom_lvl_to_be_noised.clone()
        if s.s_jitter_origin > 0.0:  # upstream's statement (inference_sampler.py:181-187)
            X_L[:, is_motif_atom_with_fixed_coord, :] += torch.normal(mean=0.0, std=s.s_jitter_origin, size=(D, 1, 3), device=X_L.device)

        X_noisy_L_traj, X_denoised_L_traj, sequence_entropy_traj, t_hats = [], [], [], []
        p_store = []
        entry = None
        outs_logits = outs_idx = None
        for step_num in range(T - 1):
            c_t_minus_1 = noise_schedule[step_num]
            c_t = noise_schedule[step_num + 1]
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            if realign:  # upstream's per-step re-centring + random rotation / translation around the motif, same arguments (inference_sampler.py:204-215)
                X_L, _ = centre_random_augment_around_motif(X_L, coord_atom_lvl_to_be_noised, is_motif_atom_with_fixed_coord, center_option=s.center_option,
                                                            centering_affects_motif=(max(step_num - 1, 0)) >= threshold_step,
                                                            s_trans=s.s_trans if step_num >= threshold_step else 0.0)
            gamma = s.gamma_0 if ns_host[step_num + 1] > s.gamma_min else 0
            step_scale = s.step_scale
            t_hat = c_t_minus_1 * (gamma + 1)
            epsilon_L = (s.noise_scale * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1)) * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=dev))
            epsilon_L[:, fixed_idx, :] = 0
            X_noisy_L = X_L + epsilon_L

            t_D = t_hat.tile(D)
            if use_graph and entry is None:
                try:
                    entry = self._get_entry(f, diffusion_module, initializer_outputs, D, L, X_noisy_L, t_D)
                except Exception as ex:  # Unsupported, or a capture failure -> eager denoiser (same arithmetic, same noise)
                    if is_oom(ex): raise  # an out-of-memory at capture is the caller's to see: never counted as a fallback, never rerouted
                    use_graph = False
                    self.stats["fallbacks"] += 1
                    GRAPH_STATS["fallbacks"] += 1
                    self.stats.setdefault("fallback_errors", []).append(f"{type(ex).__name__}: {str(ex)[:300]}")
                    logger.warning(f"[RFD3_CUDAGRAPH] {type(ex).__name__}: {ex}; eager denoiser for this design (noise order unchanged)")
                    self.entry = None
                    _set_phase(None)
                    _STATE["entry"] = None
            if use_graph:
                entry.x_in.copy_(X_noisy_L)
                entry.t_in.copy_(t_D)
                entry.graph.replay()
                X_denoised_L = entry.x_out.clone()
                outs_logits = entry.logits_out.clone()
                outs_idx = entry.idx_out.clone()
            else:
                outs = diffusion_module(X_noisy_L=X_noisy_L, t=t_D, f=f, n_recycle=s.n_recycle, **initializer_outputs)
                X_denoised_L = outs["X_L"]
                outs_logits = outs["sequence_logits_I"]
                outs_idx = outs["sequence_indices_I"]

            delta_L = (X_noisy_L - X_denoised_L) / t_hat
            d_t = c_t - t_hat
            p_store.append(torch.softmax(outs_logits, dim=-1))  # the entropy read-back is deferred to after the loop (one device->host copy, stock values)
            X_L = X_noisy_L + step_scale * d_t * delta_L
            X_noisy_L_scaled = s.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + s.sigma_data**2)
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if entry is not None and bool(entry.dup_acc.item()):
            raise AssertionError("Tensor has duplicate elements along the last dimension (deferred attention-index check).")
        if realign and torch.any(is_motif_atom_with_fixed_coord):  # upstream's closing motif re-insertion + alignment to the input motif (inference_sampler.py:338-352)
            X_L, _ = centre_random_augment_around_motif(X_L, coord_atom_lvl_to_be_noised, is_motif_atom_with_fixed_coord, reinsert_motif=s.insert_motif_at_end)
            X_L = weighted_rigid_align(coord_atom_lvl_to_be_noised, X_L, X_exists_L=is_motif_atom_with_fixed_coord)
        for p in p_store:
            p = p.cpu()
            sequence_entropy_traj.append(-torch.sum(p * torch.log(p + 1e-10), dim=-1))
        if entry is not None and entry.static_f is not None and "attn_indices" in entry.static_f:
            f["attn_indices"] = entry.static_f["attn_indices"]
        return dict(X_L=X_L, X_noisy_L_traj=X_noisy_L_traj, X_denoised_L_traj=X_denoised_L_traj, t_hats=t_hats,
                    sequence_logits_I=outs_logits, sequence_indices_I=outs_idx, sequence_entropy_traj=sequence_entropy_traj)


def install(conditional_sampler):
    """Wrap the stock 'default' sampler held by a ConditionalDiffusionSampler (called from its __init__ when enabled)."""
    stock = conditional_sampler.sampler
    if type(stock).__name__ != "SampleDiffusionWithMotif":
        logger.warning(f"[RFD3_CUDAGRAPH] sampler kind {type(stock).__name__} is not supported; graph rollout disabled")
        return conditional_sampler
    conditional_sampler.sampler = SampleDiffusionWithMotifGraph(stock)
    GRAPH_STATS["installs"] += 1
    logger.info("[RFD3_CUDAGRAPH] graph rollout enabled for the default sampler")
    return conditional_sampler
