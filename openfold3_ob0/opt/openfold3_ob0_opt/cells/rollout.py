"""The `rollout_bf16` lever: the diffusion denoiser step under bf16 autocast, the modules that read or write atom coordinates kept fp32, above a token count.

Upstream runs the whole rollout — 200 steps of the denoiser (`DiffusionModule`: conditioning, atom-attention encoder, the 24-block token
diffusion transformer, atom-attention decoder) over the sample batch plus the sampler's own coordinate arithmetic (`SampleDiffusion`: the
per-step random rotation `pos_centered @ rots^T`, the noise injection, the update `x = x_noisy + step_scale * dt * (x_noisy - x_denoised) / t`) —
inside `torch.amp.autocast(device_type=..., dtype=torch.float32)` with `use_high_precision_attention=True` (projects/of3_all_atom/model.py,
`_rollout`): every matmul fp32 (TF32 where torch allows it) whatever precision the trunk ran under. This lever draws two boundaries:

  bf16 REGION = the denoiser STEP CALL as the sampler makes it: `DiffusionModule.__call__` (a class-wide forward wrapper) and, when the
    fast-inference kit's CUDA-graph lever routes the step through `of3_graphs.GraphedStep`, that object's `__call__` too (wrapped once at the first
    sampler call, when every add-on's import-time patch is in place) — so the add-ons' eager per-rollout work inside the step call, i.e. the
    trunk-kernels pair cache refreshing its static token-pair buffers (zij [N,N,128] + 24 per-block biases [16,N,N]), runs in the same bf16
    context as the modules whose outputs it caches and holds them in bf16. In the region: the conditioning, the 24 transformer blocks (the
    `dit_attn` lever's kernel takes bf16 operands), the token-level projections — matmul-class ops bf16 with fp32 accumulation, LayerNorm /
    softmax statistics fp32 (autocast's op lists); the diffusion transformer's attention core keeps upstream's `use_high_precision` word (fp32
    softmax and fp32 P.V inside `primitives.attention._attention`) — only the projections around it move to bf16.
  fp32 ISLANDS inside it = `DiffusionModule.atom_attn_enc` and `DiffusionModule.atom_attn_dec` (instance forward pre/post hooks entering
    `torch.autocast("cuda", dtype=torch.float32)`, upstream's own rollout idiom: 16-bit inputs cast up, every matmul fp32/TF32): the encoder's
    `NoisyPositionEmbedder.linear_r` projects the scaled noisy positions and its atom transformer carries them, the decoder's `linear_q_out` emits
    the position update — so `x_out = c_skip * x_noisy + c_out * r_update` combines fp32 tensors (elementwise, no autocast op), and the sampler's
    rotation / noise / update arithmetic stays outside the region in upstream's fp32 context.

(bf16 has an 8-bit mantissa: a coordinate tens of Å from the centre cast to bf16 moves by 0.06–0.25 Å, and the rotation matmul under a
sampler-wide autocast would so round the coordinates themselves at every step — bond-length noise of that size with the fold intact. Nothing in the
bf16 region sees a coordinate.) Inside the CUDA-graphed step the Python side runs at warm-up and capture; replays repeat the captured kernels.
Numerics class: tier 2 (not bitwise with the fp32 rollout; bond-length statistics at the fp32 rollout's level).

Token gate (available, off by default): the lever serves every prediction unless the caller sets OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS; then a
prediction whose token count (the padded token dimension of the model's input batch, `batch["token_mask"].shape[-1]`, read at the model's forward)
is below it runs upstream's fp32 rollout throughout and the lever says so by name — a gate by design (census `gated=<n>`), not a fallback
(`fallbacks()` stays empty). (With the dit cells riding the bf16 region the bf16 sampler is the one to run at every size, hence no gate by
default; on upstream's attention cores alone autocast's casts would cost more than the token-level GEMMs gain at small sizes.) A call whose
token count the lever could not read is
served and counted `unsized=<n>`. The gate is decided per sampler call (`SampleDiffusion.forward`): inside a served call the step calls enter the
bf16 region and the islands; inside a gated one neither is entered.

Switches: OPENFOLD3_OB0_OPT_ROLLOUT=bf16 (exported by the `fast` line; unset = upstream's fp32 rollout), OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS=<int>
(the gate's threshold, default 0 = no gate). Lines at stderr: `[openfold3_ob0-opt/rollout] installed …`, the first served call, the first gated call, the
first bf16 step, and the exit census `LEVER name=rollout_bf16 state=on|off calls=<n> served=<n> gated=<n> unsized=<n> min_tokens=<m>
bf16_regions=<n> fp32_islands=<n> graphed_step=<0|1> dtype=bfloat16`.
"""
import atexit
import os
import sys

PREFIX = "[openfold3_ob0-opt/rollout]"
ENV = "OPENFOLD3_OB0_OPT_ROLLOUT"
ENV_MIN_TOKENS = "OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS"
MIN_TOKENS_DEFAULT = 0                                       # no gate: with the dit cells on the bf16 region the bf16 sampler serves every size; a positive
                                                             # OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS hands smaller inputs to the fp32 roll-out, counted `gated=`
FP32_ISLANDS = ("atom_attn_enc", "atom_attn_dec")          # DiffusionModule attributes kept fp32 (structure/diffusion_module.py): they read / write atom coordinates
GRAPHS_MODULE, GRAPHS_STEP = "of3_graphs", "GraphedStep"     # the fast-inference kit's graphed denoiser step (opt/forward/fast_inference/of3_levers/of3_graphs.py)
STATE = {"installed": False, "calls": 0, "served": 0, "gated": 0, "unsized": 0, "denoiser_calls": 0, "regions": 0, "islands": 0, "graphed_step": 0,
         "dtype": None, "min_tokens": None, "n_tokens": None, "classes": None, "logged": set()}
_CTXS = []                                                   # one entry per sampler call in flight: the gate's word for it ("n_tokens=800 >= 600" / "unsized"), or None when gated
_REGIONS = []                                                # one entry per bf16 entry in flight: the autocast context, or None when not serving
_ISLANDS = []                                                # one entry per fp32-island call in flight: the autocast(float32) context, or None when not serving


def _log(msg):
    print(f"{PREFIX} {msg}", file=sys.stderr, flush=True)


def _log_once(key, msg):
    if key not in STATE["logged"]:
        STATE["logged"].add(key); _log(msg)


def min_tokens(environ=None) -> int:
    """The gate's threshold: OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS (a non-negative integer), default 0 (no gate)."""
    v = ((environ if environ is not None else os.environ).get(ENV_MIN_TOKENS) or "").strip()
    if not v:
        return MIN_TOKENS_DEFAULT
    if not v.isdigit():
        raise ValueError(f"{PREFIX} {ENV_MIN_TOKENS}={v!r} is not a non-negative integer")
    return int(v)


def requested(environ=None) -> bool:
    v = ((environ if environ is not None else os.environ).get(ENV) or "").strip()
    min_tokens(environ)                                      # a mistyped threshold is refused whether or not the lever is on
    if not v:
        return False
    if v != "bf16":
        raise ValueError(f"{PREFIX} {ENV}={v!r} is not bf16 (the one value)")
    return True


def fallbacks() -> dict:
    """The lever has no per-call fallback path (autocast contexts around the step, or by the token gate none): always {} — present so the
    kit's EXIT census reads every cell lever the same way. The gate is reported on the census line (`gated=`), not here."""
    return {}


def census_line() -> str:
    return (f"{PREFIX} LEVER name=rollout_bf16 state={'on' if STATE['installed'] else 'off'} calls={STATE['calls']} served={STATE['served']} "
            f"gated={STATE['gated']} unsized={STATE['unsized']} min_tokens={STATE['min_tokens']} bf16_regions={STATE['regions']} "
            f"fp32_islands={STATE['islands']} graphed_step={STATE['graphed_step']} dtype={STATE['dtype']}")


def observe_model_call(args) -> None:
    """The model's forward pre-hook: remember the padded token count of the batch about to be predicted (None when it cannot be read)."""
    n = None
    try:
        batch = args[0] if args else None
        if isinstance(batch, dict) and "token_mask" in batch:
            n = int(batch["token_mask"].shape[-1])
    except Exception:                                        # noqa: BLE001 — an unreadable batch leaves the count unknown; the sampler call is then served and counted unsized
        n = None
    STATE["n_tokens"] = n


def gate_decision():
    """(serve: bool, why: str) for the sampler call about to run, from the last observed token count and the threshold."""
    n, m = STATE["n_tokens"], STATE["min_tokens"]
    if n is None:
        return True, "unsized"
    if n < m:
        return False, f"n_tokens={n} < {m}"
    return True, f"n_tokens={n} >= {m}"


def sampler_pre(torch=None):
    """The sampler's forward pre-hook body: count the call, decide the gate, mark the call served (its step calls enter the bf16 region and the
    fp32 islands) or gated (they stay in upstream's fp32 context throughout)."""
    STATE["calls"] += 1
    serve, why = gate_decision()
    if not serve:
        STATE["gated"] += 1; _CTXS.append(None)
        _log_once("gated", f"rollout_bf16: {why} ({ENV_MIN_TOKENS}) -> upstream's fp32 rollout for this prediction, by design")
        return
    if why == "unsized":
        STATE["unsized"] += 1
    STATE["served"] += 1
    _CTXS.append(why)
    _log_once("served", f"first served sampler call ({why})")


def sampler_post():
    """The sampler's forward post-hook body: unmark the call."""
    if _CTXS:
        _CTXS.pop()


def serving() -> bool:
    """True inside a served sampler call (the innermost one in flight); False inside a gated one or outside any."""
    return bool(_CTXS) and _CTXS[-1] is not None


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
    if getattr(denoiser, "_openfold3_ob0_opt_rollout_bf16", False):
        return 0
    n = 0
    for name in FP32_ISLANDS:
        sub = getattr(denoiser, name)
        sub.register_forward_pre_hook(lambda m, a: island_enter(torch, bool(serving())))
        sub.register_forward_hook(lambda m, a, o: island_leave(), always_call=True)
        n += 1
    denoiser._openfold3_ob0_opt_rollout_bf16 = True
    STATE["islands"] += n
    return n


def wrap_graphed_step(torch, serving=lambda: True, modules=None) -> bool:
    """Put the fast-inference kit's `GraphedStep.__call__` (the sampler's route to the denoiser under OF3_CUDA_GRAPHS=1) under the bf16 region,
    once, class-wide: the add-ons' eager per-rollout work inside it (the pair cache's static-buffer refresh) then runs and allocates in bf16 like
    the modules it stands in for. No such module loaded (graphs off) -> False, nothing to wrap: the eager step is DiffusionModule.__call__."""
    mod = (modules if modules is not None else sys.modules).get(GRAPHS_MODULE)
    cls = getattr(mod, GRAPHS_STEP, None) if mod is not None else None
    if cls is None or getattr(cls, "_openfold3_ob0_opt_rollout_bf16", False):
        return False
    inner = cls.__call__

    def __call__(self, *a, **kw):
        region_enter(torch, bool(serving()))
        try:
            return inner(self, *a, **kw)
        finally:
            region_leave()

    cls.__call__ = __call__
    cls._openfold3_ob0_opt_rollout_bf16 = True
    STATE["graphed_step"] = 1
    return True


def install(environ=None) -> dict:
    if STATE["installed"] or not requested(environ):
        return STATE
    import torch
    from openfold3.core.model.structure.diffusion_module import DiffusionModule, SampleDiffusion
    from openfold3.projects.of3_all_atom.model import OpenFold3
    STATE["min_tokens"] = min_tokens(environ); STATE["classes"] = (OpenFold3, SampleDiffusion, DiffusionModule)

    # Class-wide forward wrappers on exactly the three classes this lever observes (was: two process-wide module hooks dispatching by isinstance on
    # every nn.Module call in the process). Each wrapper makes the statements the hooks made, at the same call, in the same order among themselves:
    # `pre` before the class's forward, `post` after it — also when forward raises (the hooks were registered always_call=True). A forward wrapper
    # runs inside Module.__call__ after the module's registered pre-hooks and before its post-hooks, where the global hooks ran before / after
    # them: no other add-on hooks these three classes' instances at the sampler boundary (of3_graphs / paircache / the cells patch methods and
    # sub-modules), so the observable order is unchanged — --det 1 outputs byte-identical to the hook form (the kit's record names the check).
    model_cls, sampler_cls, denoiser_cls = STATE["classes"]

    def _wrap(cls, pre, post, tag):
        inner = cls.forward
        if getattr(inner, "_openfold3_ob0_opt_rollout_wrap", None) == tag:                 # idempotent per class
            return

        def forward(self, *args, **kwargs):
            pre(self, args)
            try:
                return inner(self, *args, **kwargs)
            finally:
                post(self)
        forward._openfold3_ob0_opt_rollout_wrap = tag; forward.__wrapped__ = inner
        forward.__name__ = getattr(inner, "__name__", "forward"); forward.__qualname__ = getattr(inner, "__qualname__", "forward"); forward.__doc__ = inner.__doc__
        cls.forward = forward

    def model_pre(module, args):
        observe_model_call(args)

    def sampler_pre_(module, args):
        wrap_graphed_step(torch, serving)                             # first sampler call: every add-on's import-time patch of the step is in place
        sampler_pre(torch)

    def denoiser_pre(module, args):
        STATE["denoiser_calls"] += 1
        attach_islands(torch, module, serving)
        region_enter(torch, serving())                                # the eager route's step boundary (inside a graphed step: a nested re-entry)

    _wrap(model_cls, model_pre, lambda m: None, "model")
    _wrap(sampler_cls, sampler_pre_, lambda m: sampler_post(), "sampler")
    _wrap(denoiser_cls, denoiser_pre, lambda m: region_leave(), "denoiser")
    STATE["installed"] = True; STATE["dtype"] = "bfloat16"; STATE["wrapped"] = [c.__module__ + "." + c.__qualname__ + ".forward" for c in (model_cls, sampler_cls, denoiser_cls)]
    _log(f"installed: for predictions of >= {STATE['min_tokens']} tokens ({ENV_MIN_TOKENS}; below it upstream's fp32 rollout throughout, by name) the "
         f"denoiser step (DiffusionModule.__call__, and {GRAPHS_MODULE}.{GRAPHS_STEP}.__call__ when the step is graphed) under torch.autocast(cuda, bfloat16) "
         f"with DiffusionModule.{' / .'.join(FP32_ISLANDS)} under torch.autocast(cuda, float32); SampleDiffusion's rotation / noise / update arithmetic keeps "
         "upstream's fp32 rollout context; attention cores keep use_high_precision")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
