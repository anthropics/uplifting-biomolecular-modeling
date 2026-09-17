"""The `dit_attn` lever: the diffusion transformer's token attention with pair bias on the core's flash kernel.

OpenFold3's sampler runs the 24-block token diffusion transformer 200 times per sampling call over the batch of samples; each block's
`AttentionPairBias` (the AdaLN kind, `use_ada_layer_norm=True`: layers/diffusion_transformer.py:81) computes softmax(q k^T / sqrt(c) + mask
bias + pair bias) v through `primitives.attention._attention` — two einsums that materialise the [S, H, N, N] logits in fp32 (the rollout runs
under upstream's own fp32 autocast, model.py). This lever serves those calls with `opt_core.kernels.dtk_kernels.flash_bias_attn` (flash-style:
the logits never reach HBM; softmax statistics in fp32 registers; operands in the sampler's dtype — bf16 under the `rollout_bf16` lever, else fp32 at the stack's TF32 matmul precision as the einsums; the pair bias read in place — the pair cache holds it head-major; a heads-last view is re-laid once per call) — or, when the core carries its
pair-bias attention entry (`opt_core.attn.apb_core`, the same flash arithmetic with the sigmoid gate left to the module: ONE launch for all samples, the
shared bias read once for all of them; `core=apb|dtk` on the census, OPENFOLD3_OPT_DIT_CORE=dtk pins the per-sample kernel), through that entry. Numerics class:
tier 2 (the fast tier's; not bitwise with the einsum path). q/k/v come from the module's own `_prep_qkv` (q pre-scaled), the gate and the output
projection from its own `_wrap_up` — only the attention core is replaced. Inside the sampler's CUDA graph (`cuda_graphs`) the Python side runs at
warm-up and capture; replays repeat the captured kernels, so the census counts captured calls, not replays.

Scope: `primitives.attention.Attention.forward` is patched class-wide and acts ONLY on instances tagged as a diffusion-transformer attention
(`AttentionPairBias.forward` tags `self.mha` on AdaLN instances) whose call is the stock eager path (no vendor kernel flag set, self-attention,
biases = [mask bias, pair bias]); every other call is the untouched stock method. Per call ONE size gate (`opt_core.attn.size_gate`;
OPENFOLD3_OPT_DIT_MIN_TOKENS, default 256) decides on N: below it the stock core runs BY DESIGN (`gated`), a shape the kernel does not take
(head dim > 128, a bias that is neither shared nor one-per-sample, a non-CUDA tensor) runs the stock core as a NAMED fallback (`fallback:<reason>`),
never silently.

Switch: OPENFOLD3_OPT_DIT=flash_bias_attn (the fast line exports it), OPENFOLD3_OPT_DIT_CORE=auto|dtk (default auto); evidence
`[openfold3-opt/dit_attn] installed ...` and one exit line `[openfold3-opt/dit_attn] LEVER name=dit_attn state=on <gate census> core=<apb|dtk(..)> core_served=<n> core_refused=<..>
modes=<core:bias_direct|dtk:bias_direct|dtk:bias_copy>:<n> core_census={the entry's own served / refused / relayout / launch counts}` (`dtk:bias_copy`: the per-sample kernel re-laid a
heads-last or other-dtype bias once per call).
"""
from __future__ import annotations

import atexit
import os
import sys
from typing import Any, Dict

from opt_core.of3_sampler import dit_rows as DR                 # the tree's row schedules + pair-bias core resolution (resolve_apb_core)

from . import apb_word                                          # the family's provider word (bound at install)

PREFIX = "[openfold3-opt/dit_attn]"
ENV = "OPENFOLD3_OPT_DIT"
ENV_MIN = "OPENFOLD3_OPT_DIT_MIN_TOKENS"
ENV_CORE = "OPENFOLD3_OPT_DIT_CORE"
MIN_DEFAULT = 256
VALUES = ("flash_bias_attn",)
CORE_VALUES = ("auto", "dtk")
KERNEL = "dtk_kernels"                                            # the core's per-sample kernel module: `dtk` on the core knob asks the provider's `dtk_loop` row (this module) by name
CORE_PREF = {"auto": "auto", "dtk": "dtk_loop"}                       # the core knob's value -> resolve_apb_core's preference (auto = the bound word, else the direct entry)
STATE: Dict[str, Any] = {"installed": False, "impl": None, "first": None, "errors": [], "core_pref": "auto", "core": None, "core_served": 0, "core_refused": {}, "modes": {}}
GATE = None
_CORE: Dict[str, Any] = {"mod": None, "tried": False, "why": ""}



def _bias_planes(pair, nb: int, H: int, N: int):
    """The DiT pair bias as the [nb, H, N, N] plane set the core's pair-bias attention takes: KEY-CONTIGUOUS (unit stride on the last axis).
    Upstream hands the bias as its [.., N, N, H] projection permuted to [.., H, N, N] -- a view whose key stride is H --; at S > 1 the core serves
    per sample and re-lays it itself, at S = 1 the fused rows read the planes directly and take unit key stride only (opt_core kernels.apb
    fpf_apb.apb_views). One `.contiguous()` then (values unchanged: bitwise; +H*N*N*itemsize transient); an already key-contiguous bias passes
    through untouched (no copy)."""
    pb = pair.reshape(nb, H, N, N)
    return pb if pb.stride(-1) == 1 else pb.contiguous()

def _apb_core():
    """The tree's pair-bias attention entry or None (dtk serves then); resolved once, the reason kept for the census."""
    if not _CORE["tried"]:
        _CORE["tried"] = True
        _CORE["mod"], _CORE["why"] = DR.resolve_apb_core(CORE_PREF.get(STATE["core_pref"], STATE["core_pref"]))
        STATE["core"] = "apb" if _CORE["mod"] is not None else "stock(%s)" % _CORE["why"]
    return _CORE["mod"]


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    c = (environ.get(ENV_CORE) or "auto").strip().lower()
    if c not in CORE_VALUES:
        raise ValueError(f"{ENV_CORE}={c!r} is not one of {'|'.join(CORE_VALUES)}")
    return True


def _core_census():
    try:
        return _CORE["mod"].census()
    except Exception:  # noqa: BLE001  (a report field; never reroutes)
        return None


def census_line() -> str:
    fields = GATE.fields() if GATE is not None else "gate=none"
    return (f"{PREFIX} LEVER name=dit_attn state={'on' if STATE['installed'] else 'off'} impl={STATE['impl']} {fields}"
            f" core={STATE['core'] or ('pending:' + STATE['core_pref'])} core_served={STATE['core_served']}"
            f" core_refused={','.join('%s:%d' % kv for kv in sorted(STATE['core_refused'].items())) or 'none'}"
            f" modes={','.join('%s:%d' % kv for kv in sorted(STATE['modes'].items())) or 'none'}"
            + (f" core_census={_core_census()}" if _CORE["mod"] is not None else "") + (f" first={STATE['first']}" if STATE["first"] else ""))


def install(environ=None) -> dict:
    """Patch Attention.forward (DiT-tagged instances only) and AttentionPairBias.forward (the tagger). Idempotent; raises by name when the
    core kernel is not importable."""
    global GATE
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    apb_word.bind(environ)                                        # the family's provider word (tier word of the line or the caller's), before the core resolves
    from opt_core.attn import size_gate
    GATE = size_gate.from_env("dit_attn", min_var=ENV_MIN, default_min=MIN_DEFAULT, environ=environ)
    STATE["core_pref"] = (environ.get(ENV_CORE) or "auto").strip().lower()
    core0 = _apb_core()                                           # resolved now: an import failure is a census word here (stock(...)), not mid-prediction
    import torch
    from openfold3.core.model.primitives.attention import Attention
    from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias

    _apb_forward = AttentionPairBias.forward

    def apb_forward(self, a, *args, **kw):
        if self.use_ada_layer_norm and not getattr(self.mha, "_of3v2_dit", False):
            self.mha._of3v2_dit = True                            # the diffusion transformer's attention (AdaLN kind): the only instances this lever serves
        return _apb_forward(self, a, *args, **kw)
    AttentionPairBias.forward = apb_forward

    _orig = Attention.forward

    def forward(self, q_x, kv_x, biases=None, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                use_triton_triangle_kernels=False, use_lma=False, **kw):
        stock = lambda: _orig(self, q_x, kv_x, biases=biases, use_deepspeed_evo_attention=use_deepspeed_evo_attention,   # noqa: E731
                              use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma, **kw)
        if not getattr(self, "_of3v2_dit", False):
            return stock()
        if use_deepspeed_evo_attention or use_cueq_triangle_kernels or use_triton_triangle_kernels or use_lma or q_x is not kv_x \
                or not q_x.is_cuda or biases is None or len(biases) != 2 or self.training:
            GATE.decide(int(q_x.shape[-2])); GATE.fallback("not_eager_selfattn_2bias")
            return stock()
        N = int(q_x.shape[-2]); H = self.no_heads; D = self.c_hidden
        if not GATE.decide(N).served:
            return stock()                                        # below the gate's floor: the stock core by design (counted `gated`)
        mask_bias, pair = biases
        lead = tuple(q_x.shape[:-2]); n = 1
        for s_ in lead:
            n *= int(s_)
        nb = 1
        for s_ in tuple(pair.shape[:-3]):
            nb *= int(s_)
        why = None
        if D > 128:
            why = f"head_dim:{D}"
        elif pair.dim() < 3 or tuple(pair.shape[-3:]) != (H, N, N) or nb not in (1, n):
            why = "bias_shape"
        elif mask_bias.shape[-1] != N or mask_bias.shape[-2] != 1 or mask_bias.numel() // N not in (1, n):
            why = "mask_shape"
        if why is not None:
            GATE.fallback(why)
            _log(f"call z-bias {tuple(pair.shape)} mask {tuple(mask_bias.shape)} a {tuple(q_x.shape)} -> the stock core (fallback:{why})") if why not in STATE["errors"] else None
            STATE["errors"].append(why)
            return stock()
        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=True)     # [*, H, N, D], q scaled by 1/sqrt(c_hidden) as the stock core expects
        qb, kb, vb = (t.reshape(n, H, N, D) for t in (q, k, v))
        pb = _bias_planes(pair, nb, H, N)                            # [nb, H, N, N], key-contiguous (S=1 reaches the core's fused rows directly)
        mb = mask_bias.reshape(-1, N)                             # rows of inf*(mask-1): 0 = keep
        nm = mb.shape[0]
        o = torch.empty((n, N, H, D), dtype=q.dtype, device=q.device)
        core = _apb_core()
        if core is None:                                              # no core to serve with (census core=stock(<reason>)): the stock core by name, counted
            GATE.fallback("core_unavailable")
            return stock()
        try:                                                          # all samples in ONE call, the shared bias read once for all of them, operands in place
            core.pair_bias_attention(qb, kb, vb, pb, (mb == 0), num_samples=n, num_heads=H, layout="shnd", out=o.view(n, N, H * D), scale=1.0)
            STATE["core_served"] += 1
            STATE["modes"]["core:bias_direct"] = STATE["modes"].get("core:bias_direct", 0) + 1     # handed to the core as held (its census says which row served and whether it re-laid a view)
        except core.Unsupported as e:                                 # refused by name before any work -> the stock core, booked
            ev = getattr(e, "event", type(e).__name__)
            STATE["core_refused"][ev] = STATE["core_refused"].get(ev, 0) + 1
            if ev not in STATE["errors"]:
                STATE["errors"].append(ev)
                _log(f"the core's pair-bias attention refused ({ev}: {e}); the stock core serves this call class")
            GATE.fallback(f"core:{ev}")
            return stock()
        if STATE["first"] is None:
            STATE["first"] = f"a{tuple(q_x.shape)}:{str(q.dtype).replace('torch.', '')}:n{n}"
            _log(f"first served call a={tuple(q_x.shape)} {q.dtype} heads={H}x{D} samples={n} bias={'shared' if nb == 1 else 'per-sample'} "
                 f"(core word={getattr(core, 'word', 'direct')} census={_core_census()})")
        return self._wrap_up(o.reshape(*lead, N, H, D), q_x)        # the module's own sigmoid gate and output projection
    Attention.forward = forward
    STATE["installed"] = True; STATE["impl"] = getattr(core0.kernel() if core0 is not None else DR, "__file__", None)
    _log(f"installed: Attention.forward serves the diffusion transformer's AttentionPairBias (AdaLN instances) with the core's pair-bias attention "
         f"({'provider word=' + core0.word if hasattr(core0, 'word') else ('direct entry' if core0 is not None else 'unavailable: ' + _CORE['why'])}) from {STATE['impl']}; {GATE.bounds_word()}")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
