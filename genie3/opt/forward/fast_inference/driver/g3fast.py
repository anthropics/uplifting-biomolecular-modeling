"""
g3fast — the resident fast-inference driver for Genie 3 (aqlaboratory/genie3 @ d77ae5ac) backbone *generation*
(binder design, unconditional, motif scaffolding requests; upstream's beam / iterative search, sequence and
side-chain prediction are outside it), carried as a LIBRARY: the batched capture line opt/genie3_opt/g3batch.py —
the one line all three modes of the kit run — imports this module unchanged for the model / sampler / data-module set-up,
the DDIM step tables and arithmetic, the CUDA-graph wrapper of the denoiser core and the process-global numerics
guard. It has no command line of its own.

Fidelity contract (NOT changed): model, weights, inputs, N=1000 / 100 DDIM sampling steps, eta, noise scale,
direction scale, binder-length distribution (same numpy draws in the same order), interface conditioning,
number of designs, PDB writer.  Only the scheduling of the computation changes:

  L1 persistent process     model + sampler + featuriser built once; many designs / problems per process
                            (stock CLI = one python process per `genie3 generate`, 20-45 s start-up each):
                            build_everything.
  L2 sync-free step         the per-step `.item()` log call, the per-step host->device timestep tensor and the
                            data-dependent `(s - step_size <= 0).all()` branch of DDIMSampler._step are replaced
                            by precomputed device tensors / a static python bool (identical values, identical
                            kernels for the math: StepTables / ddim_math) + the five value-identical forward
                            patches in g3fast_patches.py (device-side index tensors, vectorised Frenet frames,
                            static cond-group count).
  L4 CUDA-graph replay      of the denoiser core (pair_transform_net + sequence_net + structure_net; the
                            embedders stay eager because rot_to_quat calls cuSOLVER eigh) — one capture per
                            batch shape, replayed for all 100 steps (GraphedDenoiser).  Same kernels as eager
                            => bit-identical.
  numerics guard            global_numerics_state / assert_global_state: every process-global switch that changes
                            kernel selection or rounding, snapshotted once and asserted unchanged at every batch
                            boundary and around every graph replay.

Determinism plumbing copied from the stock CLI: lightning seed_everything(seed) before model construction,
torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8 (Trainer(deterministic=True)),
parameters keep requires_grad=True (at::matmul's Linear folding heuristic depends on it), inference_mode.
The request YAML is the stock experiment YAML; only its generation section is used.
"""
from __future__ import annotations

import contextlib
import os
import time
from collections import OrderedDict
from typing import Dict

import torch


# ------------------------------------------------------------------------------------------------
# set-up identical to the stock CLI path (genie3.generation.workflow._run_sample_main + GenieRunner.setup)
# ------------------------------------------------------------------------------------------------
def lightning_equivalent_determinism():
    """Stock runs under lightning.Trainer(deterministic=True), which executes
    torch.use_deterministic_algorithms(True) and sets CUBLAS_WORKSPACE_CONFIG=:4096:8 before the model reaches
    the GPU.  Both change cuBLAS/ATen kernel selection, so the driver does exactly the same (else last-bit drift)."""
    torch.use_deterministic_algorithms(True)
    os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def build_everything(config_path: str, device: torch.device, seed_override=None, outdir=None, shard_id: int = 0, num_shards: int = 1):
    """Upstream's model / diffusion / sampler / dataset construction; ``shard_id`` / ``num_shards`` slice the request exactly as upstream's
    `generate --shard-id K --num-shards M` does (config.loader.to_generation_config: the shard's share of every problem's n_sample, the designs
    named from its sample_index_offset)."""
    lightning_equivalent_determinism()
    from genie3.config import load_experiment_config, to_generation_config
    from genie3.generation.config.registry import build_sample_config_from_dict
    from genie3.generation.model.registry import get_model
    from genie3.generation.diffusion.registry import get_diffusion
    from genie3.generation.diffusion.sampler.registry import get_diffusion_sampler
    from genie3.generation.data.data_module import GenieDataModule

    exp = load_experiment_config(config_path)
    gen_dict = to_generation_config(exp, shard_id=int(shard_id), num_shards=int(num_shards))
    if outdir is not None:
        gen_dict.setdefault("io", {})["outdir"] = outdir
    config = build_sample_config_from_dict(gen_dict)
    seed = exp.experiment.seed if seed_override is None else seed_override

    # Stock order of RNG-relevant events: seed_everything(seed) -> GenieRunner() [model construction consumes
    # the global numpy RNG through scipy truncnorm inits] -> checkpoint load -> dataset featurization per design
    # [binder length via np.random.randint] -> sampling [torch CUDA default generator].
    if seed is not None:
        from lightning import seed_everything
        seed_everything(seed, workers=True)
    # Trainer(deterministic=True) side effects relevant to kernel selection
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

    t0 = time.perf_counter()
    mcfg = config.model
    model = get_model(mcfg.model)
    diffusion = get_diffusion(mcfg.diffusion)
    ck = torch.load(config.base.checkpoint, map_location="cpu")["state_dict"]
    sd = OrderedDict()
    for k, v in ck.items():
        k2 = k.replace("_orig_mod.", "").replace(".linear_motif_template.", ".linear_cond_template.")
        if k2.startswith("model."):
            sd[k2[len("model."):]] = v
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    # NOTE: parameters keep requires_grad=True exactly as in the stock CLI: at::matmul's folding heuristic
    # (bmm vs folded mm for nn.Linear(bias=False) on 3-D/4-D inputs) reads weight.requires_grad, so flipping it
    # changes GEMM kernel selection and hence last-bit numerics.  inference_mode() prevents autograd recording.
    diffusion.setup(device)
    sampler = get_diffusion_sampler(config.inference.sampler)
    sampler.set_n_timestep(diffusion.n_timestep)
    sampler.set_betas(diffusion.betas)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - t0

    dm = GenieDataModule(config.dataset)
    dm.setup("test")
    return exp, config, model, diffusion, sampler, dm, setup_s, seed


# ------------------------------------------------------------------------------------------------
# DDIM step tables (same formulas and op order as genie3/generation/diffusion/sampler/ddim.py::_step)
# ------------------------------------------------------------------------------------------------
class StepTables:
    def __init__(self, sampler, device):
        self.n_timestep = int(sampler.n_timestep)
        self.n_sample_step = int(sampler.n_sample_step)
        self.step_size = self.n_timestep // self.n_sample_step
        # stock: steps = reversed(torch.arange(1, n_timestep + 1, step_size))
        self.steps = [int(v) for v in reversed(torch.arange(1, self.n_timestep + 1, self.step_size).tolist())]
        self.is_last = [(s - self.step_size) <= 0 for s in self.steps]
        self.S = [torch.Tensor([s]).int().to(device) for s in self.steps]        # exactly stock's construction, once
        self.direction_scale = sampler.direction_scale
        self.noise_scale = sampler.noise_scale
        self.eta = sampler.eta
        self.sampler = sampler


def ddim_math(tab: StepTables, k: int, s_vec: torch.Tensor, xs, out_xl, gt_atom_mask, noise):
    """Identical tensor expression sequence to stock _step (per-sample fp32 elementwise => batch-invariant)."""
    sp = tab.sampler
    zs_out = xs - out_xl
    xs_out = ((xs - sp.sqrt_one_minus_alphas_cumprod[s_vec].view(-1, 1, 1) * zs_out)
              / sp.sqrt_alphas_cumprod[s_vec].view(-1, 1, 1)) * gt_atom_mask.unsqueeze(-1)
    if tab.is_last[k]:
        return xs_out
    t = s_vec - tab.step_size
    sigma = torch.sqrt((sp.one_minus_alphas_cumprod[t] / sp.one_minus_alphas_cumprod[s_vec])
                       * (1.0 - sp.alphas_cumprod[s_vec] / sp.alphas_cumprod[t])) * sp.eta
    direction = torch.sqrt(1 - sp.alphas_cumprod[t] - sigma ** 2).view(-1, 1, 1) * zs_out
    xt = (sp.sqrt_alphas_cumprod[t].view(-1, 1, 1) * xs_out
          + sp.direction_scale * direction
          + sp.noise_scale * sigma.view(-1, 1, 1) * noise)
    return xt


# ------------------------------------------------------------------------------------------------
# CUDA-graph wrapper of the denoiser forward at fixed (B, n_token)
# ------------------------------------------------------------------------------------------------
class GraphedDenoiser:
    """Denoiser forward = eager pre-part + CUDA-graph replay of the core.

    pre  (eager, every step): compute_noisy_structure_frames, single_feature_net, pair_feature_net.  The pair
         embedder calls torch.linalg.eigh (cuSOLVER batched Jacobi, rot_to_quat) which cannot be captured.
    core (captured once per bucket-batch, replayed every step): pair_transform_net (5 pair-transform layers),
         sequence_net, structure_net (8 IPA layers) = ~90 % of the kernels and of the GPU time.
    Activation checkpointing in LatentTransformer is switched off for the capture (at inference it is the identity
    wrapper around the same function; its RNG-state bookkeeping is what is not capturable)."""

    def __init__(self, model, batch: Dict[str, torch.Tensor], n_timestep: int, pool=None, warmup=2, compile: str = None, wide: bool = False):
        """compile  None | "inductor" (L16, mode fast): torch.compile(mode="default", dynamic=False) of the core; the COMPILED callable is what
                 the kit's own CUDA graph records (inductor's cudagraphs stay off). Dynamo traces and inductor compiles during the eager warm-up on
                 the side stream, never inside the capture, and the capture runs under the `fail_on_recompile` stance so a retrace (which would read
                 the CUDA generator's seed — illegal while capturing) is a named error instead. The TriangleMultiplicativeUpdate forward (mode fast's
                 fused-kernel adapter, or the module's own) is kept OUT of dynamo (torch._dynamo.disable): its per-call census integers would
                 specialise and retrace per call; its launches are still recorded by the graph. Numerics class 2-3: inductor fuses the elementwise /
                 LayerNorm / mask / bias / residual chains between the GEMMs and re-associates fp32 reductions — not bit-identical to the eager core,
                 never on the exact line. Cost: one compile per captured batch shape per process (tens of seconds cold, seconds with a warm
                 inductor cache), paid inside the first batch of that shape before the capture.
        wide     L18 (both kit modes): the hoisted pair featuriser's per-step TAIL runs inside the captured region and the hoist's
                 loop-invariant terms are graph-owned buffers (g3lean.wide_static_terms / wide_tail): only the noisy frames, the single
                 features and the pairwise quaternions stay eager; the pre-padded pair tensor is handed to the pair transform by
                 ownership (needs g3lean.install_lt_release — the driver's --lean-pair). A kept graph re-pointed at a new batch
                 (g3cap.rebind_graph) marks the buffers dirty and the next call refills them IN PLACE (their addresses are the graph's)."""
        from genie3.generation.utils.affine_utils import T as _T
        self._T = _T
        self.model = model
        self.batch = batch
        self.n_timestep = n_timestep
        B, N = int(batch["gt_atom_positions"].shape[0]), int(batch["gt_atom_positions"].shape[1])
        dev = batch["gt_atom_positions"].device
        ptn = model.pair_transform_net
        self._ckpt = getattr(ptn, "enable_activation_checkpointing", False)
        # A captured graph freezes the kernels selected under the numerics mode at capture time; record that
        # mode and refuse to replay under a different one (the driver never toggles mode inside a process, this is
        # the guard that proves it).  Graphs are private to one process (the line's graph cache) and freed at exit, so
        # no graph/pool outlives a request; there is no compiled-executable or feature cache keyed on shape.
        self._mode_at_capture = global_numerics_state()
        self.compile = compile
        self.wide = bool(wide)
        self._st_dirty = False
        math = self._core_math_wide if self.wide else self._core_math
        self._core_fn = self._compiled_core(compile, math) if compile else math
        with torch.inference_mode():
            x0 = torch.zeros((B, N, 3), device=dev)
            t0 = torch.ones((B,), dtype=torch.int32, device=dev)
            if self.wide:
                import g3lean                                             # opt/forward/g3cap/g3lean.py (the driver puts the g3cap directory on sys.path beside this one)
                if not g3lean.lt_release_installed():
                    raise RuntimeError("GraphedDenoiser(wide=True) needs the ownership-passing pair transform (g3lean.install_lt_release, the driver's --lean-pair)")
                self.st = g3lean.wide_static_terms(model.pair_feature_net, batch)   # the hoist's loop-invariant terms as GRAPH-OWNED buffers (refilled in place on rebind)
                batch["_g3cap_static"] = self.st
                fi, si_init, q = self._pre_head(x0, t0)
                self.rots = fi.rots.clone(); self.trans = fi.trans.clone()
                self.si_init = si_init.clone(); self.q = q.clone(); self.zij_init = None
                del fi, si_init, q
            else:
                fi, si_init, zij_init = self._pre(x0, t0)
                self.rots = fi.rots.clone(); self.trans = fi.trans.clone()
                self.si_init = si_init.clone(); self.zij_init = zij_init.clone()
            ptn.enable_activation_checkpointing = False
            try:
                self.graph = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(warmup):                  # with compile: dynamo traces + inductor compiles HERE (eager side stream), never inside the capture
                        self._core()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                stance = torch.compiler.set_stance("fail_on_recompile") if compile else contextlib.nullcontext()
                with stance, torch.cuda.graph(self.graph, pool=pool):
                    self.out_xl = self._core()
            finally:
                ptn.enable_activation_checkpointing = self._ckpt

    def _pre(self, xl, t_int):
        from genie3.generation.utils import geo_utils
        fi = geo_utils.compute_noisy_structure_frames(batch=self.batch, xl=xl)
        si_init = self.model.single_feature_net(batch=self.batch, t=t_int / self.n_timestep)
        zij_init = self.model.pair_feature_net(batch=self.batch, fi=fi, si=si_init)
        return fi, si_init, zij_init

    def _compiled_core(self, backend: str, math):
        """L16: the core as an inductor-compiled callable (see __init__). Only `inductor` is carried."""
        if backend != "inductor":
            raise ValueError(f"GraphedDenoiser(compile={backend!r}): only 'inductor' is carried")
        import torch._dynamo
        from genie3.generation.model.module.triangular_multiplicative_update import TriangleMultiplicativeUpdate as _TMU
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        if not getattr(_TMU.forward, "_g3fast_dynamo_disabled", False):      # the TriMul provider runs eagerly between compiled regions (its census ints would specialise ->
            _TMU.forward = torch._dynamo.disable(_TMU.forward)                  # a retrace per call, fatal inside a capture); its kernels are still recorded by the graph
            _TMU.forward._g3fast_dynamo_disabled = True
        return torch.compile(math, mode="default", dynamic=False, fullgraph=False)

    def _core_math(self, rots, trans, si_init, zij_init):
        fi = self._T(rots, trans)
        si, zij = self.model.pair_transform_net(si=si_init, zij=zij_init, mask=self.batch["token_mask"])
        self.model.sequence_net(si=si)                       # computed (and unused) exactly as in stock forward
        return self.model.structure_net(batch=self.batch, fi=fi, si_init=si_init, si=si, zij=zij)

    def _core_math_wide(self, rots, trans, si_init, q):
        """L18: the featuriser tail inside the captured region, then the core; the pre-padded pair tensor travels by ownership."""
        import g3lean
        fi = self._T(rots, trans)
        ptn = self.model.pair_transform_net
        zij_in = [g3lean.wide_tail(self.model.pair_feature_net, self.st, si_init, trans, q, ptn.n_global_token), True]
        si, zij = ptn(si=si_init, zij=zij_in, mask=self.batch["token_mask"])   # the pre-padded pair tensor is handed over (freed inside block 1)
        del zij_in
        self.model.sequence_net(si=si)                       # computed (and unused) exactly as in stock forward
        return self.model.structure_net(batch=self.batch, fi=fi, si_init=si_init, si=si, zij=zij)

    def _pre_head(self, xl, t_int):
        """L18: the eager head of the pre-part — noisy structure frames, single features, and the pairwise quaternions (rot_to_quat's
        batched eigen-solve synchronises with the host and cannot be captured; everything after it is)."""
        from genie3.generation.utils import geo_utils
        fi = geo_utils.compute_noisy_structure_frames(batch=self.batch, xl=xl)
        si_init = self.model.single_feature_net(batch=self.batch, t=t_int / self.n_timestep)
        q = self.model.pair_feature_net._encode_orientations(fi.rots)
        return fi, si_init, q

    def _core(self):
        if self.wide:
            return self._core_fn(self.rots, self.trans, self.si_init, self.q)
        return self._core_fn(self.rots, self.trans, self.si_init, self.zij_init)

    def __call__(self, xl, t_int):
        assert_global_state(self._mode_at_capture, "CUDA-graph replay (mode differs from capture)")
        if self.wide:
            if self._st_dirty:                                            # a kept graph re-pointed at a new batch: the loop-invariant terms recomputed INTO the graph's buffers
                import g3lean
                g3lean.wide_static_terms(self.model.pair_feature_net, self.batch, out=self.st)
                self.batch["_g3cap_static"] = self.st
                self._st_dirty = False
            fi, si_init, q = self._pre_head(xl, t_int)
            self.rots.copy_(fi.rots); self.trans.copy_(fi.trans)
            self.si_init.copy_(si_init); self.q.copy_(q)
            del fi, si_init, q
            self.graph.replay()
            return self.out_xl
        fi, si_init, zij_init = self._pre(xl, t_int)
        self.rots.copy_(fi.rots); self.trans.copy_(fi.trans)
        self.si_init.copy_(si_init); self.zij_init.copy_(zij_init)
        self.graph.replay()
        return self.out_xl


# ------------------------------------------------------------------------------------------------
# Process-global numerics state must be identical at every chained-step boundary (design / batch boundary)
# ------------------------------------------------------------------------------------------------
def global_numerics_state() -> dict:
    """Snapshot of every process-global switch that changes kernel selection or rounding."""
    st = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda.matmul.allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
        "autocast_gpu_enabled": torch.is_autocast_enabled("cuda") if callable(getattr(torch, "is_autocast_enabled", None)) else None,
        "autocast_cpu_enabled": torch.is_autocast_enabled("cpu") if callable(getattr(torch, "is_autocast_enabled", None)) else None,
        "grad_enabled": torch.is_grad_enabled(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
    }
    return st


def assert_global_state(ref: dict, where: str):
    cur = global_numerics_state()
    diff = {k: (ref[k], cur[k]) for k in ref if ref[k] != cur[k]}
    if diff:
        raise RuntimeError(f"[D59] process-global numerics state changed at {where}: {diff}")
