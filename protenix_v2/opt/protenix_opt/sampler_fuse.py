"""The diffusion transformer's fused elementwise kernels (lever ``sampler_fuse``, strategy LOCAL.dit_fused_kernels): the DIT_FUSE
add-on's F kernels (``opt/forward/DIT_FUSE/tools/dit_fuse.py``) installed on the model the stock runner builds — one Triton launch for
each of four elementwise chains of every DiffusionTransformer block instead of the stock's two or three:

* ``ada``    the AdaptiveLayerNorm tail ``sigmoid(linear_s(s)) * LN(a) + linear_nobias_s(s)`` (token and atom blocks),
* ``gate``   the attention output gating ``o * sigmoid(g)`` with the head transpose folded in (token blocks' global attention),
* ``res``    the block's output gate + residual ``sigmoid(linear_a_last(s)) * x + a``,
* ``swiglu`` the conditioned transition's ``silu(a1(a)) * a2(a)``.

Numerics: bitwise vs the stock chains — fp32 (the sampler runs with autocast disabled), libdevice ``expf`` and IEEE division as ATen's CUDA
kernels compute them, no FMA contraction (``dit_fuse.py`` docstring). LayerNorms, GEMMs and SDPA stay
stock. Its class per mode is stated as for every lever; the atom blocks' local-attention gate stays on the
stock path by construction, and so does the token blocks' gate on the non-zero ranks of the multi-GPU line (the row-sharded
gate operand is not a fused-kernel site): the stock elementwise chain runs there — the same numerics, a stock branch, not a degraded one. The
add-on's own exit line counts those calls under its word ``fallback``; this module names them ``stock_chain_calls`` (state, LEVER line, record).

Seam: ``runner.inference.InferenceRunner.__init__`` through the package's runner seam (``runner_seam``: one patch, ordered installers) — after the stock constructor returns (``init_model``, where the CLI-path sampler graph
+ DiT hoist of lever ``sampler_graph`` install when their switches are set, then ``load_checkpoint``) the add-on's
``dit_fuse_patch.install(runner.model, levers="ada,gate,res,swiglu")`` runs on the built model: the hoist's AdaLN modules keep the hoist's
forward (the add-on skips them by name), every other site gets the fused kernel; under the deterministic recipe (no sampler graph, no hoist)
every site is fused and the sampler stays eager. An install error propagates (the run stops with the add-on's traceback: no stock fallback).

Switch: ``PTX_SAMPLER_FUSE=1`` — exported by every kit mode (``modes.PACKAGE_POST``: the exact and fast rows, big through its base); ``0`` /
unset leaves the lever off (named in the report); any other value is refused by name; set under a mode whose row does not list the lever it is
refused by name (``stack._apply``). Under ``big`` the sampler is the stock eager loop (no sampler graph) and the kernels replace launches the
graph would otherwise replay: the lever's share of the sampler's time is largest there and under the deterministic recipe. big's memory
levers patch ``AttentionPairBias.standard_multihead_attention`` at the class, which the fused block forward calls: they compose; big's
Cost: none in device memory (the kernels allocate their outputs as the stock chains do); the first denoiser call of a process
compiles four Triton kernels into the box's Triton cache.

Precision regime: the fused kernels serve the fp32 diffusion sampler — stock's own rule runs ``sample_diffusion`` with autocast disabled
(``runner.inference.update_inference_configs``: ``skip_amp.sample_diffusion`` True) for every item protenix-v2 accepts (N_token <= 2560), and
big's guard lift keeps that setting above the guard at every size (``runner_hooks``: the runner's generic N_token > 3840 bf16-sampler policy is
set back), so every kit mode runs the sampler fp32. The add-on's sites (its instance-level ``forward`` / ``_wrap_up`` replacements) are written
for the fp32 sampler; this module gates each of them on the regime at call time (``torch.is_autocast_enabled``): outside autocast the
add-on's site runs (the fused kernels, bytes unchanged), and a call that arrives under autocast runs the module's own stock method (stock's
code, stock's dtypes), counted per site kind as ``regime_stock_calls`` (``ada`` AdaLN forwards, ``gate`` attention wrap-ups, ``block``
transformer-block forwards) — zero on every kit mode's run.
Beneath that, every call of the add-on's four kernel entry points (``dit_fuse.ada_tail``, ``res_gate``, ``swiglu``, ``attn_gate``) is keyed on
the kernels' own input signature — every operand fp32 and contiguous (``dit_fuse._flat``): in signature the fused kernel runs, out of signature
the stock elementwise expression runs in the operands' dtypes (``offsig_calls``; zero when the site gate holds) — never an assert, never a refusal.

Evidence: the kit's one ``LEVER name=LOCAL.dit_fused_kernels … lever=sampler_fuse`` line at exit carries ``models`` (runners the add-on
was installed on), the add-on's site counts (``ada``, ``gate``, ``res``, ``swiglu`` modules patched), its fused call counts per kernel
(``calls_*``, ``fused_calls``), the site calls the regime gate sent to the stock methods (``regime_stock_*``, ``regime_stock_calls``), the kernel
calls keyed out of signature (``offsig_calls``) and the process's ``regime`` word: ``fp32`` (no stock-method call), ``bf16`` (no fused call),
``fp32+bf16`` (items of both regimes in one process).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from . import _core  # noqa: F401  (makes the pinned opt_core importable on the no-install route)
from .report import TAG

ENV = "PTX_SAMPLER_FUSE"                     # 1: the lever is on; 0 / unset: off (named); anything else refused by name
TARGET = "runner.inference"
CLASS, METHOD = "InferenceRunner", "__init__"
STRATEGY = "LOCAL.dit_fused_kernels"         # the canonical strategy id the kit's LEVER line carries (report.STRATEGY_IDS; STRATEGIES.json)
MARK = "DIT_FUSE:"                           # the applied-marker family (stack.MARKERS)
LEVERS = "ada,gate,res,swiglu"               # the add-on's F kernels, all four (dit_fuse_patch.install levers=)
KIT = "DIT_FUSE"                             # kits.KITS key: opt/forward/DIT_FUSE, the add-on's carried tree
TOOLS = "tools"                              # the add-on directory holding dit_fuse.py / dit_fuse_patch.py (a sys.path entry while installing)
_STATE: Dict[str, Any] = {"on": False, "patch": None, "patched": False, "models": 0, "report": None, "error": None}
KERNELS = ("ada_tail", "res_gate", "swiglu", "attn_gate")   # the add-on's kernel entry points (dit_fuse), each keyed by regime_dispatch
KINDS = ("ada", "gate", "res", "swiglu")                    # the add-on's own call-count keys (dit_fuse.STATS["calls"])
SITE_KINDS = ("ada", "gate", "block")                       # the add-on's site replacements: AdaLN forward, Attention._wrap_up, DiffusionTransformerBlock forward
SITE_MODULE = "dit_fuse_patch"                             # the module whose functions the add-on puts on the instances (the hoist's AdaLN sites are another module's: kept)
_REGIME: Dict[str, Any] = {"keyed": False, "gated": 0, "stock_calls": {k: 0 for k in SITE_KINDS}, "offsig_calls": {k: 0 for k in KINDS}}


def from_env(environ=None) -> bool:
    """Whether PTX_SAMPLER_FUSE switches the lever on ("1"); "0" / unset / empty = off; any other value is refused by name."""
    v = (environ if environ is not None else os.environ).get(ENV)
    if v is None or v == "" or v == "0":
        return False
    if v == "1":
        return True
    raise ValueError(f"{ENV}={v!r}: 1 (fused diffusion-transformer elementwise kernels on) or 0 / unset")


def tools_dir() -> str:
    """opt/forward/DIT_FUSE/tools — the directory whose ``dit_fuse_patch`` / ``dit_fuse`` modules the install imports."""
    from . import kits
    return os.path.join(kits.kit_dir(KIT), TOOLS)


def _import_patch_module():
    """The add-on's ``dit_fuse_patch`` module, imported with its directory on sys.path (it imports ``dit_fuse`` by bare name). A missing
    tree or an import error (no triton libdevice, no CUDA torch) raises by name — the lever never degrades to the stock chains silently."""
    d = tools_dir()
    if not os.path.isfile(os.path.join(d, "dit_fuse_patch.py")):
        raise RuntimeError(f"sampler_fuse: {d}/dit_fuse_patch.py missing — the DIT_FUSE add-on tree is not carried at {os.path.dirname(d)}")
    if d not in sys.path:
        sys.path.insert(0, d)
    import dit_fuse_patch  # noqa: E402  (the add-on's own module; imports torch, triton, protenix)
    return dit_fuse_patch


def install_on(runner) -> None:
    """The lever's installer on the runner seam (``runner_seam``): the add-on installs on the built ``runner.model`` — after the stock
    constructor (``init_model``, where the sampler graph + DiT hoist install, then ``load_checkpoint``), before the installers registered
    after this one (the sampler attention levers)."""
    mod = _import_patch_module()
    try:
        rep = mod.install(runner.model, levers=LEVERS)
        key_kernels(kernel_module(mod))               # the signature key on the add-on's kernel entry points (the sites call them per forward, later)
        _REGIME["gated"] += gate_sites(mod._dm(runner.model))   # the regime gate on every site the add-on put on the diffusion module's instances
    except Exception as e:                       # named on the kit's stream, then raised: the run stops here (no stock fallback)
        _STATE["error"] = repr(e)
        sys.stderr.write(f"{TAG} sampler_fuse: dit_fuse_patch.install failed: {e!r}\n"); sys.stderr.flush()
        raise
    _STATE["models"] += 1; _STATE["patched"] = True
    _STATE["report"] = {k: v for k, v in dict(rep).items() if k != "cfg"}
    sys.stderr.write(f"{TAG} sampler_fuse: installed levers={LEVERS} sites={_sites(_STATE['report'])}\n"); sys.stderr.flush()


def in_signature(*tensors) -> bool:
    """The fused kernels' own input signature (``dit_fuse._flat``): every operand fp32 and contiguous (``None`` operands are absent ones)."""
    import torch
    return all(t is None or (t.dtype == torch.float32 and t.is_contiguous()) for t in tensors)


def _stock_ada_tail(a_n, x1, x2):
    import torch
    return torch.sigmoid(x1) * a_n + x2                                       # AdaptiveLayerNorm's tail: sigmoid(linear_s(s)) * LN(a) + linear_nobias_s(s)


def _stock_res_gate(gl, x, res=None, _kind="res"):
    import torch
    out = torch.sigmoid(gl) * x                                               # the output gate; + the block residual when given
    return out + res if res is not None else out


def _stock_swiglu(x, y):
    import torch.nn.functional as F
    return F.silu(x) * y                                                      # the conditioned transition's SwiGLU


def _stock_attn_gate(o, g_logits):
    import torch
    H, C = o.shape[-3], o.shape[-1]                                           # o: [..., H, N, C] (head-major); g_logits: [..., N, H*C]
    og = o.transpose(-2, -3) * torch.sigmoid(g_logits).view(g_logits.shape[:-1] + (H, C))
    return og.reshape(og.shape[:-2] + (H * C,))                               # stock: (o * sigmoid(linear_g(q))) flattened over heads


STOCK_EXPRESSIONS = {"ada_tail": ("ada", _stock_ada_tail), "res_gate": ("res", _stock_res_gate), "swiglu": ("swiglu", _stock_swiglu), "attn_gate": ("gate", _stock_attn_gate)}


def regime_dispatch(name: str, fused):
    """The keyed entry point for kernel `name`: operands in the kernels' signature -> `fused` (the add-on's kernel, unchanged); otherwise the
    stock expression in the operands' dtypes, counted under the add-on's call-count key of that kernel (``res_gate`` counts as ``gate`` when
    the add-on calls it for the attention gate: its ``_kind``)."""
    kind0, stock = STOCK_EXPRESSIONS[name]

    def entry(*args, **kwargs):
        if in_signature(*[a for a in args if a is not None and hasattr(a, "dtype")]):
            return fused(*args, **kwargs)
        kind = kwargs.get("_kind", kind0) if name == "res_gate" else kind0
        _REGIME["offsig_calls"][kind] = _REGIME["offsig_calls"].get(kind, 0) + 1
        return stock(*args, **kwargs)

    entry.__name__ = f"{name}_keyed"; entry.__wrapped__ = fused; entry._sampler_fuse_keyed = True
    return entry


def kernel_module(patch_module):
    """The add-on's kernel module ``dit_fuse`` as its site module imported it (``import dit_fuse as DF``); a site module without it is named."""
    df = getattr(patch_module, "DF", None) or sys.modules.get("dit_fuse")
    if df is None or any(not callable(getattr(df, n, None)) for n in KERNELS):
        raise RuntimeError(f"sampler_fuse: the add-on's kernel module dit_fuse with entry points {KERNELS} is not importable from {getattr(patch_module, '__file__', patch_module)}")
    return df


def key_kernels(df) -> bool:
    """Put the regime dispatch on the add-on module's four kernel entry points (``dit_fuse_patch`` calls them as module attributes). Idempotent;
    returns True when this call keyed them."""
    if all(getattr(getattr(df, n, None), "_sampler_fuse_keyed", False) for n in KERNELS):
        _REGIME["keyed"] = True; return False
    for n in KERNELS:
        f = getattr(df, n)
        if not getattr(f, "_sampler_fuse_keyed", False):
            setattr(df, n, regime_dispatch(n, f))
    _REGIME["keyed"] = True
    return True


def bf16_regime() -> bool:
    """Whether this call runs under autocast (a bf16 diffusion sampler: the runner's generic rule for N_token > 3840, which no kit mode applies — guard_lift keeps the sampler fp32)."""
    import torch
    try:
        return bool(torch.is_autocast_enabled("cuda"))
    except TypeError:                                                         # torch < 2.4: the CUDA query takes no device argument
        return bool(torch.is_autocast_enabled())


def regime_gate(kind: str, site_fn, stock_fn):
    """A site the add-on put on an instance, gated on the regime per call: fp32 sampler -> the add-on's site (`site_fn`); bf16 sampler ->
    the module's own stock method (`stock_fn`, bound to the instance), counted under `kind`."""

    def site(*args, **kwargs):
        if bf16_regime():
            _REGIME["stock_calls"][kind] = _REGIME["stock_calls"].get(kind, 0) + 1
            return stock_fn(*args, **kwargs)
        return site_fn(*args, **kwargs)

    site.__wrapped__ = site_fn; site._sampler_fuse_gated = kind
    return site


SITE_ATTRS = {"forward": None, "_wrap_up": "gate"}                              # instance attributes the add-on sets; forward's kind is by module class


def gate_sites(diffusion_module) -> int:
    """Gate every add-on site on the diffusion module's instances (attributes ``forward`` / ``_wrap_up`` holding a `dit_fuse_patch` function):
    each becomes regime_gate(kind, the add-on's function, the class's own method bound to the instance). Returns the number of sites gated."""
    n = 0
    for m in diffusion_module.modules():
        for attr, kind in SITE_ATTRS.items():
            fn = m.__dict__.get(attr)
            if fn is None or getattr(fn, "_sampler_fuse_gated", None) or getattr(fn, "__module__", None) != SITE_MODULE:
                continue
            k = kind or ("ada" if type(m).__name__ == "AdaptiveLayerNorm" else "block")
            stock = getattr(type(m), attr).__get__(m, type(m))                     # the class's method: stock's code (and any class-level lever another kit line put there)
            m.__dict__[attr] = regime_gate(k, fn, stock); n += 1
    return n


def regime_stock_calls() -> Dict[str, int]:
    """Per site kind, the calls the regime gate sent to the stock methods in this process (the bf16 regime's items)."""
    return {k: int(_REGIME["stock_calls"].get(k, 0)) for k in SITE_KINDS}


def offsig_calls() -> Dict[str, int]:
    """Per kernel, the calls the signature key sent to the stock expressions (operands out of the kernels' signature; zero while the site gate holds)."""
    return {k: int(_REGIME["offsig_calls"].get(k, 0)) for k in KINDS}


def regime_word(fused: Dict[str, int], stock: Dict[str, int]) -> str:
    """``fp32``: no stock-method / stock-expression call; ``bf16``: such calls and no fused call; ``fp32+bf16``: both (items of both regimes)."""
    f, s_ = sum(fused.values()), sum(stock.values())
    if s_ == 0:
        return "fp32"
    return "bf16" if f == 0 else "fp32+bf16"


def _sites(rep: Optional[dict]) -> Dict[str, int]:
    rep = rep or {}
    return {"ada": int(rep.get("n_ada") or 0), "gate": int(rep.get("n_gate") or 0), "res": int(rep.get("n_res_apb") or 0), "swiglu": int(rep.get("n_ctb") or 0)}


def install() -> str:
    """Arm the lever: its installer registered on the runner seam (``runner_seam``: ONE patch of ``InferenceRunner.__init__`` for every
    lever that installs on the built runner, run in registration order — this one first) and the seam patched now if ``runner.inference``
    is imported, else at its import (the core's ``patch_attr_at_import``: fail-closed, a missing seam is the kit's NOT ACTIVE line and
    exit 3). Returns the applied marker ``DIT_FUSE:<levers>(patched|armed)``."""
    from . import runner_seam
    _STATE["on"] = True
    runner_seam.add("sampler_fuse", install_on)
    _STATE["patch"] = runner_seam.arm()
    return f"{MARK}{LEVERS}({'patched' if _STATE['patch'].state == 'installed' else 'armed'})"


def calls() -> Dict[str, int]:
    """The add-on's per-kernel call counts in this process (``dit_fuse.STATS['calls']``; empty before the add-on is imported)."""
    df = sys.modules.get("dit_fuse")
    c = (getattr(df, "STATS", {}) or {}).get("calls") if df is not None else None
    return {k: int(v) for k, v in dict(c or {}).items()}


def stock_chain_calls() -> int:
    """Gate calls that ran the stock elementwise chain by construction (the atom blocks' local-attention gate; under the multi-GPU line the
    token blocks' row-sharded gate on non-zero ranks) — the count the add-on's exit line prints under its own word ``fallback``."""
    df = sys.modules.get("dit_fuse")
    v = (getattr(df, "STATS", {}) or {}).get("fallback") if df is not None else None
    return int(v or 0)


def evidence() -> list:
    """The (key, value) pairs the lever's LEVER line carries after impl/origin: runners installed on, sites patched per kernel, calls per kernel,
    gate calls on the stock chain."""
    st = state(); s = st["sites"]; c = st["calls"]; r = st["regime_stock_calls"]
    return ([("models", st["models"])] + [(k, s[k]) for k in ("ada", "gate", "res", "swiglu")] + [(f"calls_{k}", c[k]) for k in sorted(c)]
            + [("stock_chain_calls", st["stock_chain_calls"]), ("fused_calls", st["fused_calls"]), ("gated_sites", st["gated_sites"])]
            + [(f"regime_stock_{k}", r[k]) for k in SITE_KINDS]
            + [("regime_stock_calls", sum(r.values())), ("offsig_calls", sum(st["offsig_calls"].values())), ("regime", st["regime"])])


def state() -> dict:
    """The lever's own end-of-run record: whether the runner class was patched, runners the add-on installed on, the add-on's site counts,
    its fused call counts per kernel, the regime dispatch's stock-expression calls per kernel, the regime word, and the install error if one was raised."""
    patch = _STATE["patch"]
    patched = _STATE["patched"] or (patch is not None and patch.state == "installed")
    c, r = calls(), regime_stock_calls()
    return {"on": _STATE["on"], "patched": patched, "models": _STATE["models"], "sites": _sites(_STATE["report"]), "report": _STATE["report"],
            "calls": c, "fused_calls": sum(c.values()), "stock_chain_calls": stock_chain_calls(), "regime_stock_calls": r, "offsig_calls": offsig_calls(),
            "regime": regime_word(c, {**r, **{"offsig_" + k: v for k, v in offsig_calls().items()}}), "gated_sites": _REGIME["gated"], "keyed": _REGIME["keyed"],
            "error": _STATE["error"], "tools": tools_dir() if _STATE["on"] else None}
