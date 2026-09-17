"""The `rollout_bf16` lever: the diffusion denoiser step under bf16 autocast, the modules that read or write atom coordinates kept fp32.

Upstream runs the whole rollout — 200 steps of the denoiser (`DiffusionModule`: conditioning, atom-attention encoder, the 24-block token
diffusion transformer, atom-attention decoder) over the sample batch plus the sampler's own coordinate arithmetic (`SampleDiffusion`: the
per-step random rotation `pos_centered @ rots^T`, the noise injection, the update `x = x_noisy + step_scale * dt * (x_noisy - x_denoised) / t`) —
inside `torch.amp.autocast(device_type="cuda", dtype=torch.float32)` (projects/of3_all_atom/model.py, `_rollout`): every matmul fp32 (TF32 where
torch allows it) whatever precision the trunk ran under. This lever draws two boundaries:

  bf16 REGION = the denoiser STEP CALL as the sampler makes it: `DiffusionModule.forward` (wrapped once, class-wide, at install) and, when the
    fast-inference kit's CUDA-graph lever routes the step through `of3_graphs.GraphedStep`, that object's `__call__` too (wrapped once at the first
    sampler call, when every add-on's import-time patch is in place) — so the add-ons' eager per-rollout work inside the step call, i.e. the
    trunk-kernels pair cache refreshing its static token-pair buffers (zij [N,N,128] + 24 per-block biases [16,N,N]), runs in the same bf16
    context as the modules whose outputs it caches and holds them in bf16. In the region: the conditioning, the 24 transformer blocks (the
    `dit_attn` lever's kernel takes bf16 operands), the token-level projections — matmul-class ops bf16 with fp32 accumulation, LayerNorm /
    softmax statistics fp32 (autocast's op lists).
  fp32 ISLANDS inside it = `DiffusionModule.atom_attn_enc` and `DiffusionModule.atom_attn_dec` (instance forward pre/post hooks entering
    `torch.autocast("cuda", dtype=torch.float32)`, upstream's own rollout idiom: 16-bit inputs cast up, every matmul fp32/TF32): the encoder's
    `NoisyPositionEmbedder.linear_r` projects the scaled noisy positions and its atom transformer carries them, the decoder's `linear_q_out` emits
    the position update — so `x_out = c_skip * x_noisy + c_out * r_update` combines fp32 tensors (elementwise, no autocast op), and the sampler's
    rotation / noise / update arithmetic stays outside the region in upstream's fp32 context.

(bf16 has an 8-bit mantissa: a coordinate tens of Å from the centre cast to bf16 moves by 0.06–0.25 Å, and the rotation matmul under a
sampler-wide autocast would so round the coordinates themselves at every step — bond-length noise of that size with the fold intact. Nothing in the
bf16 region sees a coordinate.) Inside the CUDA-graphed step the Python side runs at warm-up and capture; replays repeat the captured kernels.
Numerics class: tier 2 (not bitwise with the fp32 rollout; bond-length statistics at the fp32 rollout's level).

Switch: OPENFOLD3_OPT_ROLLOUT=bf16 (exported by the `fast` line; unset = upstream's fp32 rollout throughout). Lines at stderr:
`[openfold3-opt/rollout] installed …`, the first bf16 step, exit census `LEVER name=rollout_bf16 state=on denoiser_calls=<n> bf16_regions=<n>
fp32_islands=<n> graphed_step=<0|1> dtype=bfloat16 hooks=class` (the boundaries are class-level wrappers of `SampleDiffusion.forward` /
`DiffusionModule.forward` and instance hooks on the two island modules — no process-wide module hooks).
"""
import atexit
import os
import sys

PREFIX = "[openfold3-opt/rollout]"
ENV = "OPENFOLD3_OPT_ROLLOUT"
FP32_ISLANDS = ("atom_attn_enc", "atom_attn_dec")          # DiffusionModule attributes kept fp32 (structure/diffusion_module.py): they read / write atom coordinates
GRAPHS_MODULE, GRAPHS_STEP = "of3_graphs", "GraphedStep"     # the fast-inference kit's graphed denoiser step (opt/forward/fast_inference/of3_levers/of3_graphs.py)
MARK = "_openfold3_opt_rollout_bf16"                          # the once-marker on wrapped classes / functions / hooked instances
STATE = {"installed": False, "denoiser_calls": 0, "regions": 0, "islands": 0, "graphed_step": 0, "dtype": None, "wrapped": []}
_REGIONS = []                                                # one entry per bf16 entry in flight: the autocast context, or None when not serving
_ISLANDS = []                                                # one entry per fp32-island call in flight: the autocast(float32) context, or None when not serving


def _log(msg):
    print(f"{PREFIX} {msg}", file=sys.stderr, flush=True)


def requested(environ=None) -> bool:
    v = ((environ if environ is not None else os.environ).get(ENV) or "").strip()
    if not v:
        return False
    if v != "bf16":
        raise ValueError(f"{PREFIX} {ENV}={v!r} is not bf16 (the one value)")
    return True


def census_line() -> str:
    return (f"{PREFIX} LEVER name=rollout_bf16 state={'on' if STATE['installed'] else 'off'} denoiser_calls={STATE['denoiser_calls']} "
            f"bf16_regions={STATE['regions']} fp32_islands={STATE['islands']} graphed_step={STATE['graphed_step']} dtype={STATE['dtype']} hooks=class")


def region_enter(torch, serve: bool = True) -> None:
    """Enter bf16 autocast for one denoiser step call (serve false: leave the call in the surrounding fp32 context)."""
    if not serve:
        _REGIONS.append(None); return
    ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    ctx.__enter__(); _REGIONS.append(ctx)
    STATE["regions"] += 1
    if STATE["regions"] == 1:
        _log(f"first bf16 step: the denoiser call under torch.autocast(cuda, bfloat16); DiffusionModule.{' / .'.join(FP32_ISLANDS)} under "
             "torch.autocast(cuda, float32); the sampler's rotation / noise / update arithmetic outside, fp32")


def region_leave() -> None:
    if _REGIONS:
        ctx = _REGIONS.pop()
        if ctx is not None:
            ctx.__exit__(None, None, None)


def island_enter(torch, serve: bool = True) -> None:
    """Enter fp32 autocast for one atom-attention encoder / decoder call (upstream's rollout idiom: 16-bit inputs cast up, matmuls fp32)."""
    if not serve:
        _ISLANDS.append(None); return
    ctx = torch.autocast("cuda", dtype=torch.float32)
    ctx.__enter__(); _ISLANDS.append(ctx)


def island_leave() -> None:
    if _ISLANDS:
        ctx = _ISLANDS.pop()
        if ctx is not None:
            ctx.__exit__(None, None, None)


def attach_islands(torch, denoiser, serving=lambda: True) -> int:
    """Register the fp32 pre/post hooks on the denoiser's FP32_ISLANDS once (idempotent per instance); `serving()` is read at each call.
    Returns the number of submodules newly hooked."""
    if getattr(denoiser, MARK, False):
        return 0
    n = 0
    for name in FP32_ISLANDS:
        sub = getattr(denoiser, name)
        sub.register_forward_pre_hook(lambda m, a: island_enter(torch, bool(serving())))
        sub.register_forward_hook(lambda m, a, o: island_leave(), always_call=True)
        n += 1
    setattr(denoiser, MARK, True)
    STATE["islands"] += n
    return n


def wrap_graphed_step(torch, serving=lambda: True, modules=None) -> bool:
    """Put the fast-inference kit's `GraphedStep.__call__` (the sampler's route to the denoiser under OF3_CUDA_GRAPHS=1) under the bf16 region,
    once, class-wide: the add-ons' eager per-rollout work inside it (the pair cache's static-buffer refresh) then runs and allocates in bf16 like
    the modules it stands in for. No such module loaded (graphs off) -> False, nothing to wrap: the eager step is DiffusionModule.__call__."""
    mod = (modules if modules is not None else sys.modules).get(GRAPHS_MODULE)
    cls = getattr(mod, GRAPHS_STEP, None) if mod is not None else None
    if cls is None or getattr(cls, MARK, False):
        return False
    inner = cls.__call__

    def __call__(self, *a, **kw):
        region_enter(torch, bool(serving()))
        try:
            return inner(self, *a, **kw)
        finally:
            region_leave()

    cls.__call__ = __call__
    setattr(cls, MARK, True)
    STATE["graphed_step"] = 1
    return True


def wrap_classes(torch, SampleDiffusion, DiffusionModule, serving=lambda: True) -> list:
    """The lever's two boundaries as CLASS-LEVEL wrappers, each installed once (idempotent; no process-wide module hooks — nothing runs on the
    calls of modules the lever does not serve): `SampleDiffusion.forward` wraps the graphed step class at its first call (every add-on's
    import-time patch of the step is in place by then); `DiffusionModule.forward` attaches the fp32 islands to the instance at its first call and
    runs the engine's forward inside the bf16 region (the eager route's step boundary; inside a graphed step a nested re-entry), left in a
    `finally`. `serving()` is read at each call. Returns the attributes newly wrapped."""
    done = []
    if not getattr(DiffusionModule.forward, MARK, False):
        inner_d = DiffusionModule.forward

        def forward(self, *a, **kw):                                  # noqa: E306 — DiffusionModule.forward under the bf16 region
            STATE["denoiser_calls"] += 1
            attach_islands(torch, self, serving)
            region_enter(torch, bool(serving()))
            try:
                return inner_d(self, *a, **kw)
            finally:
                region_leave()
        setattr(forward, MARK, True); forward.__wrapped__ = inner_d
        DiffusionModule.forward = forward; done.append("DiffusionModule.forward")
    if not getattr(SampleDiffusion.forward, MARK, False):
        inner_s = SampleDiffusion.forward

        def forward(self, *a, **kw):                                  # noqa: E306 — SampleDiffusion.forward: the graphed step class wrapped once, then the engine's sampler
            wrap_graphed_step(torch, serving)
            return inner_s(self, *a, **kw)
        setattr(forward, MARK, True); forward.__wrapped__ = inner_s
        SampleDiffusion.forward = forward; done.append("SampleDiffusion.forward")
    return done


def install(environ=None) -> dict:
    if STATE["installed"] or not requested(environ):
        return STATE
    import torch
    from openfold3.core.model.structure.diffusion_module import DiffusionModule, SampleDiffusion
    STATE["wrapped"] = wrap_classes(torch, SampleDiffusion, DiffusionModule)
    STATE["installed"] = True; STATE["dtype"] = "bfloat16"
    _log(f"installed: the denoiser step (DiffusionModule.forward, wrapped class-wide, and {GRAPHS_MODULE}.{GRAPHS_STEP}.__call__ when the step is graphed) under "
         f"torch.autocast(cuda, bfloat16) with DiffusionModule.{' / .'.join(FP32_ISLANDS)} under torch.autocast(cuda, float32); "
         "SampleDiffusion's rotation / noise / update arithmetic keeps upstream's fp32 rollout context; no process-wide module hooks")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
