"""The `dit_attn` lever: the diffusion transformer's token attention with pair bias on the core's flash kernel, on the 16-bit operands of the bf16 roll-out.

OpenFold3's sampler runs the 24-block token diffusion transformer 200 times per sampling call over the batch of samples; each block's
`DiffusionAttentionPairBias` (the AdaLN attention class of openfold3 >= 0.5.0, layers/attention_pair_bias.py, built by the token blocks of
layers/diffusion_transformer.py; the atom transformer's `CrossAttentionPairBias` is another class and is never touched) computes
softmax(q k^T / sqrt(c) + mask bias + pair bias) v through `primitives.attention._attention` — two einsums that materialise the [S, H, N, N]
logits, in fp32: upstream asks `use_high_precision_attention=True` for the roll-out (projects/of3_all_atom/model.py: fp32 logits, softmax and
P.V whatever the sampler's precision). This lever serves those calls with `opt_core.kernels.dtk_kernels.flash_bias_attn` (flash-style: the
logits never reach HBM; softmax statistics in fp32 registers; q k^T and P.V on bf16 tensor cores with fp32 accumulation; one [H, N, D] problem
per sample, the pair bias shared by the samples read in place — the pair cache holds it head-major; a heads-last view is re-laid once per call —, or, when the core
carries its pair-bias attention entry `opt_core.attn.apb_core`, ONE launch of it for all samples with the shared bias read once for all of them: `core=apb|dtk` on the
census, OPENFOLD3_OB0_OPT_DIT_CORE=dtk pins the per-sample kernel) WHEN the sampler's operands are 16-bit — under the
`rollout_bf16` lever (the roll-out under bf16 autocast; every prediction unless the caller sets OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS). An fp32
roll-out (below a caller's gate, or `rollout_bf16` off) keeps upstream's attention core call for call: the fused kernel serves bf16 operands only,
so fp32 operands are GATED by dtype, by design, and counted (`gated_by=…,dtype_fp32:<n>`). The operand dtype is the one
upstream's `Linear` will give q/k/v (`projected_dtype`: a 16-bit input, or autocast on for the input's device), decided before any projection
runs. Upstream's high-precision word is therefore NOT honoured on a served call (its operands are 16-bit), and the census counts exactly those
calls (`high_precision_asked=<n> high_precision_overridden=<n>`): the lever's one departure from upstream's statement, named, never silent.
Numerics class: tier 2 (the fast tier's; not bitwise with the
einsum path). q/k/v come from the module's own `_prep_qkv` (q pre-scaled), the gate and the output projection from its own `_wrap_up` — only
the attention core is replaced. Inside the sampler's CUDA graph (`cuda_graphs`) the Python side runs at warm-up and capture; replays repeat the
captured kernels, so the census counts captured calls, not replays.

Scope: `primitives.attention.Attention.forward` is patched class-wide and acts ONLY on instances tagged as a diffusion-transformer attention
(`DiffusionAttentionPairBias.forward` tags `self.mha`) whose call is the stock eager path (no vendor kernel flag set, self-attention, biases =
[mask bias, pair bias]); every other call is the untouched stock method. Per call the size gate (`opt_core.attn.size_gate`;
OPENFOLD3_OB0_OPT_DIT_MIN_TOKENS, default 256) decides on N and the dtype gate on the operands: below the floor or on fp32 operands the stock
core runs BY DESIGN (`gated`, `gated_by=lt_min:<n>,dtype_fp32:<n>`), a shape the kernel does not take
(head dim > 128, a bias that is neither shared nor one-per-sample, a non-CUDA tensor) runs the stock core as a NAMED fallback
(`fallback:<reason>`, and `dit_attn:<reason>=<n>` in the kit's exit tally through `fallbacks()`), never silently.

Switch: OPENFOLD3_OB0_OPT_DIT=flash_bias_attn (the `fast` line exports it; absent = upstream's attention core as shipped), OPENFOLD3_OB0_OPT_DIT_CORE=auto|dtk
(default auto). Evidence
`[openfold3_ob0-opt/dit_attn] installed ...` and one exit line `[openfold3_ob0-opt/dit_attn] LEVER name=dit_attn state=on impl=<file>
gate.dit_attn=min256,dtype16 calls=<n> served=<n> gated=<n> fallback=<n> [fallback_by=…] gated_by=lt_min:<n>,dtype_fp32:<n> high_precision_asked=<n>
high_precision_overridden=<n> core=<apb|dtk(..)> core_served=<n> core_refused=<..> modes=<core:bias_direct|dtk:bias_direct|dtk:bias_copy>:<n>
core_census={the entry's own served / refused / relayout / launch counts} [first=...]` — calls = served + gated + fallback; `dtk:bias_copy` counts the
per-sample kernel's once-per-call relayout of a heads-last or other-dtype bias.
"""
from __future__ import annotations

import atexit
import os
import sys
from typing import Any, Dict

from opt_core.of3_sampler import dit_rows as DR                 # the tree's row schedules + pair-bias core resolution (resolve_apb_core)

from . import apb_word                                          # the family's provider word (bound at install)

PREFIX = "[openfold3_ob0-opt/dit_attn]"
ENV = "OPENFOLD3_OB0_OPT_DIT"
ENV_MIN = "OPENFOLD3_OB0_OPT_DIT_MIN_TOKENS"
ENV_CORE = "OPENFOLD3_OB0_OPT_DIT_CORE"
MIN_DEFAULT = 256
VALUES = ("flash_bias_attn",)
CORE_VALUES = ("auto", "dtk")
DTYPE_WORD = "dtype16"                                              # the dtype gate's word on the LEVER line: 16-bit operands are served, fp32 operands run upstream's core (gated_by dtype_fp32)
KERNEL = "dtk_kernels"                                            # the core's per-sample kernel module: `dtk` on the core knob asks the provider's `dtk_loop` row (this module) by name
CORE_PREF = {"auto": "auto", "dtk": "dtk_loop"}                       # the core knob's value -> resolve_apb_core's preference (auto = the bound word, else the direct entry)
STATE: Dict[str, Any] = {"installed": False, "impl": None, "first": None, "first_gated_dtype": None, "errors": [], "hp_asked": 0, "hp_overridden": 0, "dtype_gated": 0,
                         "core_pref": "auto", "core": None, "core_served": 0, "core_refused": {}, "modes": {}}
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


def projected_dtype(x, linear):
    """The dtype upstream's `primitives.linear.Linear` gives `linear(x)` — decided the way its forward decides it, computing nothing: a Linear
    with a set `precision` returns x's dtype, a bf16 input is projected in bf16, and otherwise `nn.functional.linear` runs in the autocast dtype
    when autocast is on for x's device, else in x's dtype. This is q/k/v's dtype before `_prep_qkv` runs (linear_q/k/v share the class)."""
    import torch
    if getattr(linear, "precision", None) is not None or x.dtype is torch.bfloat16:
        return x.dtype
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        return torch.get_autocast_dtype(dev)
    return x.dtype


def serves_dtype(dtype) -> bool:
    """The dtype gate: the kernel serves 16-bit operands (bf16 / fp16); fp32 operands run upstream's attention core, by design."""
    import torch
    return dtype in (torch.bfloat16, torch.float16)


def gate_pairs():
    """The gate census as ordered (key, value) pairs — the size gate's own (`opt_core.attn.size_gate.SizeGate.evidence`) with the dtype gate
    folded in: its word joins the bounds word (`min256,dtype16`), its count joins `calls` and `gated`, and `gated_by` names both routes by design
    (`lt_min`: below the size floor; `dtype_fp32`: fp32 operands). calls == served + gated + fallback."""
    c = GATE.census(); dg = int(STATE["dtype_gated"])
    pairs = [("gate.dit_attn", f"{GATE.bounds_word()},{DTYPE_WORD}"), ("calls", c["calls"] + dg), ("served", c["served"]), ("gated", c["gated"] + dg), ("fallback", c["fallback"])]
    if c["fallback_by"]:
        pairs.append(("fallback_by", dict(sorted(c["fallback_by"].items()))))
    pairs.append(("gated_by", {"lt_min": int(c["gated_by"].get("lt_min", 0)), "dtype_fp32": dg}))
    return pairs


def fallbacks() -> Dict[str, int]:
    """{'<reason>': n} over the NAMED fallbacks counted so far ({} = none; the size gate's `gated` calls are a route by design, not here): what
    the kit's EXIT line aggregates as `fallbacks=dit_attn:<reason>=<n>`."""
    if GATE is None:
        return {}
    return {reason: n for reason, n in sorted(GATE.census()["fallback_by"].items()) if n}


def _core_census():
    try:
        return _CORE["mod"].census()
    except Exception:  # noqa: BLE001  (a report field; never reroutes)
        return None


def census_line() -> str:
    if GATE is None:
        fields = "gate=none"
    else:
        from opt_core import report
        fields = report.kv(*gate_pairs())
    return (f"{PREFIX} LEVER name=dit_attn state={'on' if STATE['installed'] else 'off'} impl={STATE['impl']} {fields} "
            f"high_precision_asked={STATE['hp_asked']} high_precision_overridden={STATE['hp_overridden']}"
            f" core={STATE['core'] or ('pending:' + STATE['core_pref'])} core_served={STATE['core_served']}"
            f" core_refused={','.join('%s:%d' % kv for kv in sorted(STATE['core_refused'].items())) or 'none'}"
            f" modes={','.join('%s:%d' % kv for kv in sorted(STATE['modes'].items())) or 'none'}"
            + (f" core_census={_core_census()}" if _CORE["mod"] is not None else "") + (f" first={STATE['first']}" if STATE["first"] else ""))


def install(environ=None) -> dict:
    """Patch Attention.forward (DiT-tagged instances only) and DiffusionAttentionPairBias.forward (the tagger). Idempotent; raises by name when
    the core kernel is not importable."""
    global GATE
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    apb_word.bind(environ)                                        # the family's provider word (tier word of the line or the caller's), before the core resolves
    STATE["core_pref"] = (environ.get(ENV_CORE) or "auto").strip().lower()
    core0 = _apb_core()                                           # resolved now: an import failure is a census word here (stock(...)), not mid-prediction
    from opt_core.attn import size_gate
    GATE = size_gate.from_env("dit_attn", min_var=ENV_MIN, default_min=MIN_DEFAULT, environ=environ)
    import torch
    from openfold3.core.model.primitives.attention import Attention, DEFAULT_LMA_KV_CHUNK_SIZE, DEFAULT_LMA_Q_CHUNK_SIZE
    from openfold3.core.model.layers.attention_pair_bias import DiffusionAttentionPairBias

    _apb_forward = DiffusionAttentionPairBias.forward

    def apb_forward(self, a, *args, **kw):
        if not getattr(self.mha, "_of3v2_dit", False):
            self.mha._of3v2_dit = True                            # the diffusion transformer's token attention (the AdaLN class): the only instances this lever serves
        return _apb_forward(self, a, *args, **kw)
    DiffusionAttentionPairBias.forward = apb_forward

    _orig = Attention.forward

    def forward(self, q_x, kv_x, biases=None, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                use_triton_triangle_kernels=False, use_lma=False, lma_q_chunk_size=DEFAULT_LMA_Q_CHUNK_SIZE,
                lma_kv_chunk_size=DEFAULT_LMA_KV_CHUNK_SIZE, use_high_precision=False):
        stock = lambda: _orig(self, q_x, kv_x, biases=biases, use_deepspeed_evo_attention=use_deepspeed_evo_attention,   # noqa: E731
                              use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma,
                              lma_q_chunk_size=lma_q_chunk_size, lma_kv_chunk_size=lma_kv_chunk_size, use_high_precision=use_high_precision)
        if not getattr(self, "_of3v2_dit", False):
            return stock()
        if use_deepspeed_evo_attention or use_cueq_triangle_kernels or use_triton_triangle_kernels or use_lma or q_x is not kv_x \
                or not q_x.is_cuda or biases is None or len(biases) != 2 or self.training:
            GATE.decide(int(q_x.shape[-2])); GATE.fallback("not_eager_selfattn_2bias")
            return stock()
        N = int(q_x.shape[-2]); H = self.no_heads; D = self.c_hidden
        if not GATE.admits(N):
            GATE.decide(N)                                        # below the size gate's floor: the stock core by design (counted `gated`, gated_by lt_min)
            return stock()
        dt = projected_dtype(q_x, self.linear_q)                  # q/k/v's dtype-to-be, decided before any projection runs
        if not serves_dtype(dt):                                  # fp32 operands (the fp32 roll-out: below the rollout_bf16 gate, or that lever off): upstream's core by design (counted `gated`, gated_by dtype_fp32)
            STATE["dtype_gated"] += 1
            if STATE["first_gated_dtype"] is None:
                STATE["first_gated_dtype"] = f"a{tuple(q_x.shape)}:{str(dt).replace('torch.', '')}"
                _log(f"first fp32-operand call a={tuple(q_x.shape)} -> upstream's attention core by design (gated_by dtype_fp32: the kernel serves 16-bit operands, the bf16 roll-out's)")
            return stock()
        GATE.decide(N)                                            # served by size and dtype (counted `served`; a shape refusal below re-books it as a named fallback)
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
        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=True)     # [*, H, N, D], q scaled by 1/sqrt(c_hidden) as the stock core expects; 16-bit (the dtype gate above)
        if use_high_precision:                                    # upstream's fp32-core word (model.py's sample_diffusion call): counted per served call, overridden (the operands are 16-bit)
            STATE["hp_asked"] += 1
            if q.dtype != torch.float32:
                STATE["hp_overridden"] += 1
        qb, kb, vb = (t.reshape(n, H, N, D) for t in (q, k, v))
        pb = _bias_planes(pair, nb, H, N)                            # [nb, H, N, N], key-contiguous (S=1 reaches the core's fused rows directly)
        mb = mask_bias.reshape(-1, N)                             # rows of inf*(mask-1): 0 = keep
        nm = mb.shape[0]
        o = torch.empty((n, N, H, D), dtype=q.dtype, device=q.device)
        core = _apb_core()
        if core is None:                                              # no core to serve with (census core=stock(<reason>)): upstream's core by name, counted
            GATE.fallback("core_unavailable")
            return stock()
        try:                                                          # all samples in ONE call, the shared bias read once for all of them, operands in place
            core.pair_bias_attention(qb, kb, vb, pb, (mb == 0), num_samples=n, num_heads=H, layout="shnd", out=o.view(n, N, H * D), scale=1.0)
            STATE["core_served"] += 1
            STATE["modes"]["core:bias_direct"] = STATE["modes"].get("core:bias_direct", 0) + 1     # handed to the core as held (its census says which row served and whether it re-laid a view)
        except core.Unsupported as e:                                 # refused by name before any work -> upstream's core, booked
            ev = getattr(e, "event", type(e).__name__)
            STATE["core_refused"][ev] = STATE["core_refused"].get(ev, 0) + 1
            if ev not in STATE["errors"]:
                STATE["errors"].append(ev)
                _log(f"the core's pair-bias attention refused ({ev}: {e}); upstream's core serves this call class")
            GATE.fallback(f"core:{ev}")
            return stock()
        if STATE["first"] is None:
            STATE["first"] = f"a{tuple(q_x.shape)}:{str(q.dtype).replace('torch.', '')}:n{n}:hp{int(bool(use_high_precision))}"
            _log(f"first served call a={tuple(q_x.shape)} {q.dtype} heads={H}x{D} samples={n} bias={'shared' if nb == 1 else 'per-sample'} "
                 f"core={getattr(core, 'word', 'direct')} census={_core_census()} "
                 f"use_high_precision={bool(use_high_precision)} ({'overridden: ' + str(q.dtype).replace('torch.', '') + ' operands' if use_high_precision and q.dtype != torch.float32 else 'held: fp32 operands' if use_high_precision else 'not asked'})")
        return self._wrap_up(o.reshape(*lead, N, H, D), q_x)        # the module's own sigmoid gate and output projection
    Attention.forward = forward
    STATE["installed"] = True; STATE["impl"] = getattr(core0.kernel() if core0 is not None else DR, "__file__", None)
    _log(f"installed: Attention.forward serves the diffusion transformer's DiffusionAttentionPairBias instances with the core's pair-bias attention "
         f"({'provider word=' + core0.word if hasattr(core0, 'word') else ('direct entry' if core0 is not None else 'unavailable: ' + _CORE['why'])}) from {STATE['impl']} "
         f"on 16-bit operands (the bf16 roll-out's; fp32 operands keep upstream's core, gated by dtype; upstream's use_high_precision word counted and overridden on served calls); {GATE.bounds_word()},{DTYPE_WORD}")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
