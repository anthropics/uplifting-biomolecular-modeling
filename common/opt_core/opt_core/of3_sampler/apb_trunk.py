"""Lever `apb_trunk` (fast class) of the OF3 code family: the single track's attention with pair bias in the Pairformer trunk (48 blocks x
(1 + recycles) passes) and the confidence head's pairformer (4 blocks per sample) — `AttentionPairBias.forward` of the trunk instances (0.4.x:
the non-AdaLN instances of the one class; 0.5.x: the class itself, the diffusion transformer having its own) — as: the module's LayerNorm of a;
the pair bias produced HEAD-MAJOR (LayerNorm(z) by the module's own `layer_norm_z`, then ONE GEMM `W_z @ LN(z)^T -> [H, N*N]`: the [*, H, N, N]
planes directly, no [N, N, H] intermediate and no permute pass); the module's q / k / v / gate projections; ONE launch of the core's pair-bias
attention (`opt_core.attn.apb_core.pair_bias_attention`, layout "shnd": the strided [*, H, N, D] projection views in place, the key mask by
name, the sigmoid gate fused, per-sample bias planes when z carries a sample dimension — the confidence head's batched samples); the
module's output projection. Stock: LayerNorm(z) -> linear_z -> permute -> four projections -> [*, H, N, N] logits materialised in the
attention core -> gate -> output projection.

Served: CUDA, no grad, self-attention through the eager core (no `use_deepspeed_evo_attention` / `use_lma` / `use_cueq_triangle_kernels` /
`use_triton_triangle_kernels` flag), the trunk under 16-bit autocast (the fast lines' `bf16-mixed`), z present with N tokens matching a, at
least ENV_MIN tokens (default 0), head dim <= 64. `use_high_precision_attention` (0.5.x asks it on every trunk call: fp32 logits, softmax and
P.V) follows the kit's HIGH_PRECISION word: "honour" runs the stock forward for such calls (counted `flag:use_high_precision_attention`),
"override" serves them on the 16-bit operands with fp32 logit accumulation and fp32 softmax statistics and COUNTS it
(`high_precision_asked / high_precision_overridden`). Every other call runs the stock forward COUNTED BY REASON (`fallback=<reason>:<n>`);
every check happens before any work. Numerics class: tolerance (the head-major bias GEMM and the flash softmax sum in a different order than
stock's; P enters the P.V product in the operand dtype).

Pair-bias producer: ENV_PRODUCER = ln_proj | mm chooses between the tree's fused LayerNorm+projection kernel (`opt_core.kernels.ln_proj`:
one read of z, the planes written head-major into 16-byte-aligned rows; a call it refuses by name takes the GEMM producer, counted as
producer_fallback) and the module's LayerNorm + one head-major GEMM; the kit adapter binds the switch name and its default (PRODUCER_DEFAULT).
`use_producer(name, fn)` lets another lever serve the bias instead of the module's LN + the
head-major GEMM (`fn(module, z, dtype) -> [*, H, N, N]` head-major, rows 16-byte aligned; anything else is still served — a heads-last view
is copied once per call inside the core, counted there); the census names the producer in use. Without a fused producer the module's
`layer_norm_z(z)` remains the dominant cost of the call (about 70 % at 2 000 tokens).

Two run-time words refine the served set (attribution knobs; defaults change nothing): ENV_HIGH_PRECISION=override|honour replaces the kit's
HIGH_PRECISION word for the process, and ENV_SCOPE=trunk+confidence|trunk — under `trunk` the calls made inside the confidence head's
pairformer (`PairformerEmbedding.forward` of M_CONF, wrapped to mark the context) run the stock forward, counted `scope:confidence`; the
census reports `scope=<word> served_confidence=<n>` either way.

A kit binds PREFIX / ENV / ENV_MIN / ENV_HIGH_PRECISION / ENV_SCOPE / CONFLICT_ENV (the trunk-kernels add-on's vendor route for the same
forward: both set is refused at activation) / HIGH_PRECISION / M_APB / M_CONF with `configure(...)` from its `cells/apb_trunk.py` adapter and
installs from its cells hook; evidence `<PREFIX> installed ...` and one exit line `<PREFIX> LEVER name=apb_trunk state=on served=<n>
served_confidence=<n> fallback=<..> producer=<name> high_precision=<word> scope=<word> high_precision_asked=<n> high_precision_overridden=<n>
core_census={...}`.
"""
from __future__ import annotations

import atexit
import os
import sys
from typing import Any, Callable, Dict, Optional

from opt_core.of3_sampler import dit_rows as DR                 # the tree's row schedules + pair-bias core resolution (resolve_apb_core)

PREFIX = "[opt_core/apb_trunk]"
ENV = "OF3_FAMILY_APB_TRUNK"                                       # rebound by the kit adapter (configure)
ENV_MIN = "OF3_FAMILY_APB_TRUNK_MIN_TOKENS"
ENV_PRODUCER = "OF3_FAMILY_APB_TRUNK_PRODUCER"                     # ln_proj | mm (rebound by the kit adapter)
PRODUCER_DEFAULT = "mm"                                            # the producer when ENV_PRODUCER is unset (the kit adapter binds its line's default)
PRODUCERS = ("ln_proj", "mm")
CONFLICT_ENV = "OF3T_APB"                                          # the trunk-kernels add-on's vendor-kernel route for the same forward
ENV_HIGH_PRECISION = "OF3_FAMILY_APB_TRUNK_HIGH_PRECISION"           # run-time word over HIGH_PRECISION (attribution knob)
ENV_SCOPE = "OF3_FAMILY_APB_TRUNK_SCOPE"                             # trunk+confidence | trunk
HIGH_PRECISION = "honour"                                          # honour | override (see the module docstring): the kit's default word
M_APB = None                                                       # the engine's attention_pair_bias module path: bound by the kit adapter
M_CONF = None                                                      # the engine's module holding CONF_CLASS (the confidence head's pairformer embedding), or None
CONF_CLASS = "PairformerEmbedding"
VALUES = ("1",)
HIGH_PRECISION_WORDS = ("honour", "override")
SCOPE_WORDS = ("trunk+confidence", "trunk")
MIN_DEFAULT = 0
CORE_MODULE = DR.APB_CORE_MODULE                                  # the tree's pair-bias attention entry (dit_rows.resolve_apb_core)
FLAGS = ("use_deepspeed_evo_attention", "use_triton_triangle_kernels", "use_lma", "use_cueq_triangle_kernels", "use_high_precision_attention")
KNOWN_KW = FLAGS + ("mask", "s")
CONFIGURABLE = ("PREFIX", "ENV", "ENV_MIN", "ENV_HIGH_PRECISION", "ENV_SCOPE", "ENV_PRODUCER", "PRODUCER_DEFAULT", "CONFLICT_ENV", "HIGH_PRECISION", "M_APB", "M_CONF")


def configure(**kw) -> None:
    """The kit adapter's binding: reassigns this module's engine words (log prefix, switch names, the conflicting add-on switch, the
    high-precision word, the engine's module path) before install; unknown names or words raise."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        if k == "HIGH_PRECISION" and v not in HIGH_PRECISION_WORDS:
            raise ValueError(f"{__name__}.configure: HIGH_PRECISION={v!r} is not one of {'|'.join(HIGH_PRECISION_WORDS)}")
        if k == "PRODUCER_DEFAULT" and v not in PRODUCERS:
            raise ValueError(f"{__name__}.configure: PRODUCER_DEFAULT={v!r} is not one of {'|'.join(PRODUCERS)}")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "impl": None, "served": 0, "served_confidence": 0, "fallback": {}, "errors": {},
                         "first": None, "min_tokens": MIN_DEFAULT, "producer": "headmajor_mm", "producer_fallback": {}, "high_precision": None, "scope": "trunk+confidence",
                         "conf_depth": 0, "high_precision_asked": 0, "high_precision_overridden": 0}
_CORE: Dict[str, Any] = {"mod": None}
_LNPROJ: Dict[str, Any] = {"mod": None}
_PRODUCER: Dict[str, Optional[Callable]] = {"fn": None}
_ONCE = set()


class Refuse(Exception):
    """A by-name refusal raised before any fused work; the stock forward runs, counted."""


def _log(msg: str, once_key=None) -> None:
    if once_key is not None:
        if once_key in _ONCE:
            return
        _ONCE.add(once_key)
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def _hp() -> str:
    """The high-precision word in force: the run-time word when the switch set one at install, else the kit's HIGH_PRECISION."""
    return STATE["high_precision"] or HIGH_PRECISION


def _words(environ) -> tuple:
    """(high-precision word, scope word) from the environment, validated by name; unset = the kit's word / trunk+confidence."""
    hp = (environ.get(ENV_HIGH_PRECISION) or "").strip() if ENV_HIGH_PRECISION else ""
    if hp and hp not in HIGH_PRECISION_WORDS:
        raise ValueError(f"{ENV_HIGH_PRECISION}={hp!r} is not one of {'|'.join(HIGH_PRECISION_WORDS)}")
    sc = (environ.get(ENV_SCOPE) or "").strip() if ENV_SCOPE else ""
    if sc and sc not in SCOPE_WORDS:
        raise ValueError(f"{ENV_SCOPE}={sc!r} is not one of {'|'.join(SCOPE_WORDS)}")
    if sc == "trunk" and not M_CONF:
        raise ValueError(f"{ENV_SCOPE}=trunk needs the confidence head's module bound (the kit adapter's configure(M_CONF=))")
    return (hp or HIGH_PRECISION), (sc or "trunk+confidence")


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    mt = (environ.get(ENV_MIN) or "").strip()
    if mt and not mt.isdigit():
        raise ValueError(f"{ENV_MIN}={mt!r} is not a non-negative integer token count")
    if (environ.get(CONFLICT_ENV) or "").strip():
        raise ValueError(f"{ENV} and {CONFLICT_ENV} both route the trunk's AttentionPairBias.forward: set one")
    _words(environ)
    pr = (environ.get(ENV_PRODUCER) or "").strip()
    if pr and pr not in PRODUCERS:
        raise ValueError(f"{ENV_PRODUCER}={pr!r} is not one of {'|'.join(PRODUCERS)}")
    return True


def use_producer(name: str, fn: Optional[Callable]) -> None:
    """Register the pair-bias producer of the served calls: fn(module, z, dtype) -> the bias [*, H, N, N] (head-major, unit key stride, rows
    16-byte aligned is the fast layout); None restores the module's LayerNorm + head-major GEMM."""
    _PRODUCER["fn"] = fn
    STATE["producer"] = name if fn is not None else "headmajor_mm"


def headmajor_bias(module, z, dtype):
    """The stock producer's statement with a head-major result: LN(z) (the module's layer_norm_z), then W_z @ LN(z)^T per leading index ->
    [*, H, N, N] planes (bf16/fp16 operands under autocast as stock's linear_z, fp32 accumulation in the GEMM)."""
    import torch
    zn = module.layer_norm_z(z)
    lead = tuple(zn.shape[:-3]); N1, N2, C = int(zn.shape[-3]), int(zn.shape[-2]), int(zn.shape[-1])
    w = module.linear_z.weight
    H = int(w.shape[0])
    zn3 = zn.reshape((-1, N1 * N2, C))
    if zn3.dtype != dtype:
        zn3 = zn3.to(dtype)
    pb = torch.matmul(w.to(dtype), zn3.transpose(1, 2))                # [B, H, N*N]
    b = getattr(module.linear_z, "bias", None)
    if b is not None:
        pb = pb + b.to(pb.dtype)[None, :, None]
    return pb.view(lead + (H, N1, N2)) if lead else pb.view(H, N1, N2)


def lnproj_bias(module, z, dtype):
    """The fused producer: the tree's ln_proj kernel reads z ONCE (any view with unit channel stride) and writes LayerNorm(z) @ W_z^T head-major
    into 16-byte-aligned rows ([*, H, N, ld][..., :N], ld = N rounded up to 8) — no LN(z) tensor, no [N, N, H] intermediate. A call the kernel
    refuses by name (width, head count, dtype, a linear_z bias, device) takes headmajor_bias for that call, counted as producer_fallback."""
    import torch
    K = _LNPROJ["mod"]
    lz, ln = module.linear_z, module.layer_norm_z
    if K is None or getattr(lz, "bias", None) is not None or dtype not in (torch.bfloat16, torch.float32) or not z.is_cuda:
        why = "no-kernel" if K is None else "linear-bias" if getattr(lz, "bias", None) is not None else f"dtype-{dtype}" if z.is_cuda else "device"
        _count(STATE["producer_fallback"], why)
        return headmajor_bias(module, z, dtype)
    key = (id(lz.weight), id(ln.weight), str(z.device), dtype == torch.float32)
    cache = module.__dict__.setdefault("_apb_lnproj", {})
    P = cache.get(key)
    if P is None:
        try:
            P = K.pack_pair_bias_weights(ln.weight.detach(), None if getattr(ln, "bias", None) is None else ln.bias.detach(), lz.weight.detach(),
                                        float(getattr(ln, "eps", 1e-5)), z.device, dot_fp32=(dtype == torch.float32))
        except K.Unsupported as e:
            _count(STATE["producer_fallback"], "pack:" + e.reason)
            return headmajor_bias(module, z, dtype)
        cache.clear(); cache[key] = P
    z4 = z if z.dim() == 4 else z.reshape((-1,) + tuple(z.shape[-3:])) if z.dim() > 4 else z.unsqueeze(0)
    try:
        pb = K.pair_bias(z4, P, out_layout="bhij", out_dtype=dtype)          # [B, H, N, ld][..., :N]
    except K.Unsupported as e:
        _count(STATE["producer_fallback"], e.reason)
        return headmajor_bias(module, z, dtype)
    lead = tuple(z.shape[:-3])
    return pb.view(lead + tuple(pb.shape[1:])) if z.dim() > 4 else pb if z.dim() == 4 else pb[0]


def _select_producer(environ) -> None:
    """ENV_PRODUCER (or PRODUCER_DEFAULT): ln_proj registers lnproj_bias unless another lever registered its own producer first; mm keeps the
    module's LN + head-major GEMM. A missing kernel module is named on the install line and the GEMM producer stays."""
    want = (environ.get(ENV_PRODUCER) or "").strip() or PRODUCER_DEFAULT
    if want != "ln_proj" or _PRODUCER["fn"] is not None:
        return
    try:
        from opt_core.kernels import route
        import importlib
        route("ln_proj")                                                   # carried kernels resolve by NAME through the core's route
        _LNPROJ["mod"] = importlib.import_module("ln_proj")
    except ImportError as e:
        _log(f"producer ln_proj unavailable ({type(e).__name__}: {e}); the module's LayerNorm + head-major GEMM produces the bias")
        return
    use_producer("ln_proj", lnproj_bias)


def _core():
    if _CORE["mod"] is None:
        C, why = DR.resolve_apb_core("auto")                       # routes the carried kernel now: an import failure is named at activation, not mid-prediction
        if C is None:
            raise RuntimeError(f"{PREFIX} {CORE_MODULE} unavailable ({why}) — set {ENV}= (empty) or repair the core install (opt/pyproject.toml pins it)")
        _CORE["mod"] = C
    return _CORE["mod"]


_PLAN: Dict[tuple, tuple] = {}
_PLAN_MAX = 1024


def plan(module, a, z, mask, flags: dict):
    """(ok, reason) for serving this call — every check before any work. The verdict is a pure function of what the checks read (the
    module object and its train/precision words, operand shapes / dtypes / device, the compute dtype, grad mode, the flags, the token gate and
    the precision policy), so it is memoised on exactly that key: a trunk's 48 blocks x recycles repeat a handful of keys."""
    import torch
    in_conf = STATE["conf_depth"] > 0
    mha = getattr(module, "mha", None)
    key = (id(module), getattr(mha, "c_hidden", None), getattr(mha, "no_heads", None), module.training, torch.is_grad_enabled(),
           getattr(getattr(module, "linear_z", None), "precision", None),
           a.shape, a.dtype, a.device, None if z is None else (z.shape, z.dtype, z.device), None if mask is None else (mask.shape, mask.device),
           DR.compute_dtype(a), tuple(sorted((k, bool(v)) for k, v in flags.items())), STATE["min_tokens"], _hp(), STATE["scope"], in_conf)
    hit = _PLAN.get(key)
    if hit is not None:
        return hit
    verdict = _plan(module, a, z, mask, flags)
    if len(_PLAN) >= _PLAN_MAX:
        _PLAN.clear()
    _PLAN[key] = verdict
    return verdict


def _plan(module, a, z, mask, flags: dict):
    import torch
    if STATE["scope"] == "trunk" and STATE["conf_depth"] > 0:
        return False, "scope:confidence"
    for k, v in flags.items():
        if v and not (k == "use_high_precision_attention" and _hp() == "override"):
            return False, "flag:" + k
    if module.training or torch.is_grad_enabled():
        return False, "grad_enabled"
    if not a.is_cuda:
        return False, "device:not_cuda"
    if z is None:
        return False, "no_pair"
    cd = DR.compute_dtype(a)
    if cd not in (torch.bfloat16, torch.float16):
        return False, "dtype:%s" % str(cd).replace("torch.", "")
    if a.dim() < 2 or z.dim() < 3:
        return False, "rank"
    N = int(a.shape[-2])
    if N < STATE["min_tokens"]:
        return False, "gated:min_tokens"
    if int(z.shape[-2]) != N or int(z.shape[-3]) != N:
        return False, "shape:z"
    mha = getattr(module, "mha", None)
    if mha is None or not hasattr(mha, "_prep_qkv") or not hasattr(mha, "linear_o") or getattr(module, "linear_z", None) is None \
            or getattr(module, "layer_norm_z", None) is None or getattr(module, "layer_norm_a", None) is None:
        return False, "module_layout"
    if getattr(module.linear_z, "precision", None) is not None:     # an explicit per-layer precision override on the bias projection: the stock Linear honours it, the head-major GEMM would not
        return False, "linear_precision"
    if int(getattr(mha, "c_hidden", 0)) > 64 or int(getattr(mha, "c_hidden", 0)) < 1:
        return False, "head_dim:%s" % getattr(mha, "c_hidden", "?")
    S = 1
    for d in a.shape[:-2]:
        S *= int(d)
    SB = 1
    for d in z.shape[:-3]:
        SB *= int(d)
    if SB not in (1, S):
        return False, "shape:z_lead"
    if mask is not None:
        if int(mask.shape[-1]) != N:
            return False, "shape:mask"
        try:
            torch.broadcast_shapes(tuple(mask.shape), tuple(a.shape[:-1]))
        except RuntimeError:
            return False, "shape:mask_lead"
    return True, ""


def served_forward(module, a, z, mask, high_precision_asked: bool = False):
    """The served statement (see the module docstring); returns the module's output [*, N, c_q]."""
    C = _core()
    mha = module.mha
    lead = tuple(a.shape[:-2]); N = int(a.shape[-2]); H = int(mha.no_heads); D = int(mha.c_hidden)
    S = 1
    for d in lead:
        S *= int(d)
    cd = DR.compute_dtype(a)
    a_ln = module.layer_norm_a(a)
    fn = _PRODUCER["fn"]
    pb = fn(module, z, cd) if fn is not None else headmajor_bias(module, z, cd)
    if tuple(pb.shape[-3:]) != (H, N, N):
        raise Refuse("producer_shape")
    pb = pb.reshape((-1, H, N, N))                                  # [SB, H, N, N], SB in {1, S} (plan checked z's lead)
    km = None
    if mask is not None:
        mlead = 1
        for d in mask.shape[:-1]:
            mlead *= int(d)
        km = mask.reshape(1, N) if mlead == 1 else mask.expand(tuple(a.shape[:-1])).reshape(S, N)
    q, k, v = mha._prep_qkv(a_ln, a_ln, apply_scale=True)          # [*, H, N, D] strided views, q scaled by 1/sqrt(D) as the stock core expects
    q4, k4, v4 = (t.reshape((S, H, N, D)) for t in (q, k, v))
    g = mha.linear_g(a_ln) if getattr(mha, "linear_g", None) is not None else None
    g3 = g.reshape((S, N, H * D)) if g is not None else None
    o = C.pair_bias_attention(q4, k4, v4, pb, km, gate=g3, num_samples=S, num_heads=H, layout="shnd", scale=1.0,
                              inf=float(getattr(module, "inf", 1e9)))                                   # [S, N, H*D]
    if high_precision_asked:                                         # the engine's fp32-core word on this call: served on the 16-bit operands (high-precision word "override"), counted
        STATE["high_precision_asked"] += 1; STATE["high_precision_overridden"] += 1
    if STATE["conf_depth"] > 0:
        STATE["served_confidence"] += 1
    if STATE["first"] is None:
        STATE["first"] = "a%s:%s:S%d:H%d:N%d:D%d:bias%s:%s:hp%d" % (tuple(a.shape), str(cd).replace("torch.", ""), S, H, N, D, tuple(pb.shape),
                                                                 "x".join(str(int(x)) for x in pb.stride()), int(bool(high_precision_asked)))
        _log("first served call a=%s %s heads=%dx%d samples=%d bias=%s strides=%s producer=%s" % (tuple(a.shape), cd, H, D, S, tuple(pb.shape),
                                                                                                    tuple(pb.stride()), STATE["producer"]))
    return mha.linear_o(o.view(lead + (N, H * D)))


def _patch(mod) -> bool:
    APB = mod.AttentionPairBias
    if getattr(APB.forward, "_of3opt_apb_trunk", False):
        return False
    orig = APB.forward

    def forward(self, a, z, *args, **kw):
        stock = lambda: orig(self, a, z, *args, **kw)              # noqa: E731
        if STATE["state"] != "on" or type(self) is not APB or getattr(self, "use_ada_layer_norm", False):
            return stock()                                          # 0.4.x AdaLN (diffusion transformer) instances and subclasses are other levers' calls
        mask = kw.get("mask")
        flags = {k: kw.get(k, False) for k in FLAGS}
        unknown = sorted(k for k in kw if k not in KNOWN_KW) + (["positional_args"] if args else []) + (["s"] if kw.get("s") is not None else [])
        try:
            ok, why = (False, "kwargs:" + ",".join(unknown)) if unknown else plan(self, a, z, mask, flags)
        except Exception as e:  # noqa: BLE001
            from ..oom import is_oom
            if is_oom(e):
                raise
            ok, why = False, "plan_error:%s" % type(e).__name__
        if not ok:
            _count(STATE["fallback"], why)
            _log("call a=%s -> the stock forward (fallback:%s)" % (tuple(a.shape), why), once_key=("fb", why))
            return stock()
        try:
            out = served_forward(self, a, z, mask, high_precision_asked=bool(flags["use_high_precision_attention"]))
            STATE["served"] += 1
            return out
        except Refuse as e:
            why = str(e) or "refused"
        except _core().Unsupported as e:                                 # the core refused this shape by name before any work
            why = "core:%s" % getattr(e, "event", "unsupported")
        except Exception as e:  # noqa: BLE001                           # anything unexpected: named, counted, the stock forward runs (the inputs are untouched); an OOM is the caller's
            from ..oom import is_oom
            if is_oom(e):
                raise
            why = "error:%s" % type(e).__name__
            _count(STATE["errors"], why)
            _log("the served path raised %r on a=%s -> the stock forward for this call (counted as %s)" % (e, tuple(a.shape), why), once_key=("err", why))
        _count(STATE["fallback"], why)
        _log("call a=%s -> the stock forward (fallback:%s)" % (tuple(a.shape), why), once_key=("fb", why))
        return stock()
    forward._of3opt_apb_trunk = True; forward.__wrapped__ = orig
    APB.forward = forward
    return True


def _patch_conf(mod) -> bool:
    """Wrap CONF_CLASS.forward of `mod` so calls inside the confidence head's pairformer are recognisable (STATE['conf_depth'])."""
    cls = getattr(mod, CONF_CLASS, None)
    if cls is None:
        raise RuntimeError(f"{PREFIX} {getattr(mod, '__name__', mod)} has no {CONF_CLASS}")
    if getattr(cls.forward, "_of3opt_apb_trunk_conf", False):
        return False
    orig = cls.forward

    def forward(self, *args, **kw):
        STATE["conf_depth"] += 1
        try:
            return orig(self, *args, **kw)
        finally:
            STATE["conf_depth"] -= 1
    forward._of3opt_apb_trunk_conf = True; forward.__wrapped__ = orig
    cls.forward = forward
    return True


def census_line() -> str:
    core_c = None
    try:
        core_c = _CORE["mod"].census() if _CORE["mod"] is not None else None
    except Exception:  # noqa: BLE001
        pass
    return (f"{PREFIX} LEVER name=apb_trunk state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" impl={STATE['impl']} min_tokens={STATE['min_tokens']} served={STATE['served']} served_confidence={STATE['served_confidence']}"
            f" fallback={','.join('%s:%d' % kv for kv in sorted(STATE['fallback'].items())) or 'none'}"
            f" producer={STATE['producer']} producer_fallback={','.join('%s:%d' % kv for kv in sorted(STATE['producer_fallback'].items())) or 'none'}"
            f" high_precision={_hp()} scope={STATE['scope']} high_precision_asked={STATE['high_precision_asked']}"
            f" high_precision_overridden={STATE['high_precision_overridden']}"
            + (f" core_census={core_c}" if core_c is not None else "") + (f" first={STATE['first']}" if STATE["first"] else ""))


def install(environ=None) -> dict:
    """Resolve the core entry + its carried kernel, patch AttentionPairBias.forward class-wide. Idempotent; raises by name when the core is not
    importable."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not M_APB:
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_APB=) before install")
    C = _core()
    STATE["min_tokens"] = int((environ.get(ENV_MIN) or "").strip() or MIN_DEFAULT)
    hp, scope = _words(environ)
    STATE["high_precision"] = hp if hp != HIGH_PRECISION else None
    STATE["scope"] = scope
    _select_producer(environ)
    import importlib
    if M_CONF:
        _patch_conf(importlib.import_module(M_CONF))                # marks the confidence head's pairformer calls (served_confidence; the `trunk` scope refuses them by name)
    _patch(importlib.import_module(M_APB))
    STATE["state"] = "on"; STATE["impl"] = getattr(C.kernel(), "__file__", None)
    STATE["installed"] = True
    _log(f"installed: AttentionPairBias.forward serves the pairformer single track's attention ({'trunk and confidence head' if scope != 'trunk' else 'trunk'} instances"
         f"{'' if scope != 'trunk' else '; confidence head calls run the stock forward, counted scope:confidence'}) "
         f"with the pair bias produced head-major and {CORE_MODULE}.pair_bias_attention from {STATE['impl']}; min_tokens={STATE['min_tokens']} "
         f"producer={STATE['producer']} high_precision={hp} scope={scope}")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
