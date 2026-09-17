"""The `dit_glue` lever: the token diffusion transformer block as ONE schedule of fused pieces (fast class).

Stock `DiffusionTransformerBlock.forward` (layers/diffusion_transformer.py; 24 blocks × 200 steps over the S samples; a [.., S, N, 768] fp32
residual stream `a`, conditioning `s` [.., 1, N, 384] — one row per token serving all samples — and the pair representation `z`) launches
~130 kernels per block: the AdaLN as an fp32 LayerNorm + the s-side LayerNorm + two GEMMs + sigmoid / mul / add, four separate q / k / v /
gate projections each re-casting the AdaLN output, per-sample mask casts, the attention core, the sigmoid gate, the output projection, the
adaLN-zero output gate (GEMM + sigmoid + mul), the residual add, and the conditioned transition's own sixteen. This lever runs the block as

    x    = AdaLN(a, s)                          dit_rows.adaln_forward: rows in the compute dtype, s rows periodic over the samples
    qkvg = x @ [W_q | W_k | W_v | W_g]^T (+ b)  ONE GEMM [S·N, c] × [c, 4·H·D] (weights packed once per module in the compute dtype)
    o    = attention(q, k, v, pair bias, key mask, gate)   q / k / v / gate = the four column slices of qkvg IN PLACE (no head transposes,
                                                no copies); the pair bias exactly as `AttentionPairBias._prep_bias` hands it (the trunk-kernels
                                                pair cache's per-block tensor under `paircache`), read in place when head-major with unit key
                                                stride in the compute dtype, else after ONE head-major relayout per call (`bias_copy`); the
                                                token-mask rows read in place. Core: the tree's pair-bias attention entry
                                                (`opt_core.attn.apb_core.pair_bias_attention`, all samples in one call, the bias read once for
                                                all S) when the core carries it, else `opt_core.kernels.dtk_kernels.flash_bias_attn` per sample —
                                                `core=apb|dtk` on the census; <KIT>_DIT_GLUE_CORE=dtk pins the latter; an entry that
                                                refuses a shape by name is booked (`core_refused`) and dtk serves that call
    u    = o @ W_o^T (+ b_o)                    ONE GEMM
    a    = a + sigmoid(s @ W_ada^T + b_ada) · u dtk_kernels.gate_residual (gate rows periodic), written onto the fp32 residual stream
    a    = a + ConditionedTransition(a, s, mask)   dit_rows.cond_transition_forward with the residual folded into its last pass

— 11 kernels per block. Every GEMM accumulates in fp32; the row kernels compute in fp32 and round once. Numerics: tier 2 (the fast line's);
the block's output error against an fp64 evaluation of the stock block is inside the stock block's own error under bf16 autocast at sampler
steps 0 / 99 / 199 (the kits' tests/test_dit_cells.py builds the comparison on random weights; the measured inputs were real 800- and
2,000-token block inputs).

Domain, refused BY NAME otherwise (the stock block runs, counted `fallback:<reason>`): self-attention blocks (`use_cross_attention` False)
with AdaLN (`use_ada_layer_norm`), eval mode, CUDA tensors, head dim ≤ 128, the stock projection layouts, no vendor-kernel flag on the call
(`flag:<name>`; the `use_high_precision_attention` word is honoured the same way unless the kit configures HIGH_PRECISION="override", when the
call is served on 16-bit operands with fp32 softmax statistics and accumulation and counted `high_precision_overridden`), a 16-bit compute
dtype (an fp32 rollout is `gated:cd_float32`, by design), no Linear precision override, the pair bias [.., H, N, N] shared or one per sample, N ≥ <KIT>_DIT_GLUE_MIN_TOKENS
(default 0: every size), and dit_rows' own AdaLN / transition domains (`transition_adaln:<why>`, `transition_cond:<why>`). A refusal
raised inside the schedule before its fused work (`bias_shape`, `adaln_out:<dtype>`) and any unexpected error (`error:<Type>`, logged once)
also run the stock block — the inputs are never mutated, a prediction never dies of this lever. Inside the sampler's CUDA graph the Python
side runs at warm-up, capture and the memo levers' eager refresh step; replays repeat the captured kernels, so the census counts captured
calls. With this lever serving, `dit_attn` (the attention-core-only lever) sees only the calls this one refuses.

The conditioned sub-layers OUTSIDE this block schedule — the sequence-local atom transformer's blocks (the sampler's atom-attention encoder
and decoder, 3 + 3 blocks × 200 steps, fp32 under the rollout's fp32 island; their attention half is `atom_window`'s), the input embedder's
atom encoder, and any token block this schedule refuses — reach `ConditionedTransitionBlock.forward` and `AdaLN.forward` as module calls.
With <KIT>_DIT_GLUE_ROWS=all (the default) those two class methods are served class-wide by the same dit_rows schedules
(`cond_transition_forward`: AdaLN rows -> ONE [a|b] GEMM -> swiglu -> GEMM -> gate·mask row kernel, 6 kernels for the stock sixteen; the caller
adds the residual; `adaln_forward`: 3 kernels for seven), in the call's own compute dtype (fp32 rows stay fp32: TF32 GEMMs like the stock
Linear under float32_matmul_precision "high", the row kernels in fp32). Refused BY NAME per call, the stock method runs, counted
(`cond:<why>` / `adaln:<why>` with dit_rows' domain words, `cond:chunked` for a chunk_size call, `adaln:mixed_<dtype>` for an AdaLN call whose
compute dtype is not its input's — stock returns its LayerNorm's dtype there and the consumer may route on it, so only same-dtype calls are
served; `…:error:<Type>` for anything unexpected). <KIT>_DIT_GLUE_ROWS=block keeps the two class methods stock (the block schedule only).

Switches (the kit adapter's names, `configure(ENV=…)`): <KIT>_DIT_GLUE=1; <KIT>_DIT_GLUE_MIN_TOKENS (default 0); <KIT>_DIT_GLUE_CORE=auto|dtk
(default auto); <KIT>_DIT_GLUE_ROWS=all|block (default all). Evidence `<PREFIX> installed …` and one exit line `<PREFIX> LEVER name=dit_glue
state=on served=<n> fallback=<..> modes=<..> core=<apb|dtk(..)> core_served=<n> core_refused=<..> high_precision=<honour|override>
high_precision_asked=<n> high_precision_overridden=<n> rows=<all|block> cond_served=<n> cond_fallback=<..> cond_modes=<..> adaln_served=<n>
adaln_fallback=<..>`.
Engines: the OF3 code family (0.4.x `AttentionPairBias(use_ada_layer_norm=True)`, the 0.5.x fork's `DiffusionAttentionPairBias`); the kits'
`cells/dit_glue.py` are the adapters (switch names, log prefix, the engine's module path M_DIT, the high-precision word).
"""
from __future__ import annotations

import atexit
import math
import os
import sys
from typing import Any, Dict, Optional

from . import dit_rows as DR

PREFIX = "[opt_core/of3_sampler.dit_glue]"               # the kit adapter names itself and its switches: configure(PREFIX=, ENV=, ENV_MIN=, ENV_CORE=, HIGH_PRECISION=)
ENV: Optional[str] = None                                # <KIT>_DIT_GLUE=1
ENV_MIN: Optional[str] = None                            # <KIT>_DIT_GLUE_MIN_TOKENS
ENV_CORE: Optional[str] = None                           # <KIT>_DIT_GLUE_CORE=auto|dtk
ENV_ROWS: Optional[str] = None                           # <KIT>_DIT_GLUE_ROWS=all|block
HIGH_PRECISION = "honour"                                # the call's `use_high_precision_attention=True` word: honour = the stock block runs (flag refusal, counted);
                                                         # override = served on 16-bit operands with fp32 softmax statistics and accumulation, counted high_precision_overridden (the 0.5.x fork asks it on every rollout call)
VALUES = ("1",)
CORE_VALUES = ("auto", "dtk")
ROWS_VALUES = ("all", "block")                           # all = the block schedule + ConditionedTransitionBlock / AdaLN served class-wide; block = the block schedule only
MIN_DEFAULT = 0
KERNEL = "dtk_kernels"                                            # the core's carried kernel module, gated by name at activation
CORE_MODULE = DR.APB_CORE_MODULE                                  # the tree's pair-bias attention entry (dit_rows.resolve_apb_core)
M_DIT: Optional[str] = None                              # the engine module holding DiffusionTransformerBlock (….core.model.layers.diffusion_transformer)
CONFIGURABLE = ("PREFIX", "ENV", "ENV_MIN", "ENV_CORE", "ENV_ROWS", "HIGH_PRECISION", "M_DIT")
def configure(**kw) -> None:
    """The kit adapter's binding: reassigns this module's engine words (log prefix, switch names, the high-precision word, the engine's module path) before install; unknown names raise."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "impl": None, "served": 0, "fallback": {}, "modes": {}, "errors": {},
                         "first": None, "min_tokens": MIN_DEFAULT, "core_pref": "auto", "core": None, "core_served": 0, "core_refused": {},
                         "high_precision_asked": 0, "high_precision_overridden": 0,
                         "rows": "all", "rows_state": "off", "cond_served": 0, "cond_fallback": {}, "cond_modes": {}, "cond_first": None,
                         "adaln_served": 0, "adaln_fallback": {}}
_DK = {"mod": None}
_CORE = {"mod": None, "tried": False, "why": ""}
_ONCE = set()


class Refuse(Exception):
    """A by-name refusal inside the schedule, raised before its fused work; the stock block runs, counted."""


def _log(msg: str, once_key=None) -> None:
    if once_key is not None:
        if once_key in _ONCE:
            return
        _ONCE.add(once_key)
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if not ENV:                                                     # not bound by a kit adapter (configure(ENV=...)): nothing requested
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    c = (environ.get(ENV_CORE) or "auto").strip().lower()
    if c not in CORE_VALUES:
        raise ValueError(f"{ENV_CORE}={c!r} is not one of {'|'.join(CORE_VALUES)}")
    mt = (environ.get(ENV_MIN) or "").strip()
    if mt and not mt.isdigit():
        raise ValueError(f"{ENV_MIN}={mt!r} is not a non-negative integer token count")
    rw = ((environ.get(ENV_ROWS) if ENV_ROWS else None) or "all").strip().lower()
    if rw not in ROWS_VALUES:
        raise ValueError(f"{ENV_ROWS}={rw!r} is not one of {'|'.join(ROWS_VALUES)}")
    return True


def _apb_core():
    """The tree's pair-bias attention entry or None (dtk serves then); resolved once, the reason kept for the census."""
    if not _CORE["tried"]:
        _CORE["tried"] = True
        _CORE["mod"], _CORE["why"] = DR.resolve_apb_core(STATE["core_pref"])
        STATE["core"] = "apb" if _CORE["mod"] is not None else "dtk(%s)" % _CORE["why"]
    return _CORE["mod"]


# ----------------------------------------------------------------------------------------------------------------- plan / weights / schedule
def plan(blk, a, s, z, mask, flags: dict, mask_trans: bool = True):
    """(ok, reason) for serving this block call on the fused schedule."""
    import torch
    if getattr(blk, "use_cross_attention", True):
        return False, "cross_attention_block"
    for k, v in flags.items():
        if v and not (k == "use_high_precision_attention" and HIGH_PRECISION == "override"):
            return False, "flag:" + k
    apb = getattr(blk, "attention_pair_bias", None); tr = getattr(blk, "conditioned_transition", None)
    if apb is None or tr is None or not hasattr(apb, "mha"):
        return False, "block_layout"
    mha = apb.mha
    ada = getattr(apb, "use_ada_layer_norm", None)                                              # 0.4.x: a flag on AttentionPairBias; the 0.5.x fork's AdaLN class (DiffusionAttentionPairBias) has no flag
    if ada is False or not hasattr(apb, "linear_ada_out") or not hasattr(getattr(apb, "layer_norm_a", None), "layer_norm_s"):
        return False, "no_adaln"
    if blk.training:
        return False, "training"
    if not (torch.is_tensor(a) and a.is_cuda and torch.is_tensor(s) and s.is_cuda and torch.is_tensor(z)):
        return False, "not_cuda"
    N = int(a.shape[-2])
    if N < STATE["min_tokens"]:
        return False, "gated:lt_min"
    if DR.compute_dtype(a) not in (torch.bfloat16, torch.float16):                               # an fp32 rollout (no bf16 region, or below its gate) keeps the stock block: gated by dtype, by design
        return False, "gated:cd_%s" % str(DR.compute_dtype(a)).replace("torch.", "")
    H, D = int(mha.no_heads), int(mha.c_hidden)
    if D > 128:
        return False, "head_dim:%d" % D
    c = int(a.shape[-1])
    for lin in (mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_o, apb.linear_ada_out) + ((mha.linear_g,) if getattr(mha, "linear_g", None) is not None else ()):
        if getattr(lin, "precision", None) is not None:
            return False, "linear_precision"
    if tuple(mha.linear_q.weight.shape) != (H * D, c) or tuple(mha.linear_k.weight.shape) != (H * D, c) or tuple(mha.linear_v.weight.shape) != (H * D, c) \
            or tuple(mha.linear_o.weight.shape) != (c, H * D):
        return False, "channels"
    if apb.linear_ada_out.bias is None or tuple(apb.linear_ada_out.weight.shape) != (c, int(s.shape[-1])):
        return False, "ada_out_layout"
    if mask is not None and int(mask.shape[-1]) != N:
        return False, "mask_shape"
    ok, why, _ = DR.plan_adaln(apb.layer_norm_a, a, s)                                          # dit_rows' own domains are this lever's too
    if not ok:
        return False, "transition_adaln:%s" % why
    ok, why, _ = DR.plan_cond_transition(tr, a, s, mask if mask_trans else None)
    if not ok:
        return False, "transition_cond:%s" % why
    return True, ""


def _weights(blk, cd, device) -> dict:
    import torch
    apb = blk.attention_pair_bias; mha = apb.mha
    lins = [mha.linear_q, mha.linear_k, mha.linear_v] + ([mha.linear_g] if getattr(mha, "linear_g", None) is not None else [])
    params = tuple(l.weight for l in lins) + tuple(l.bias for l in lins if l.bias is not None) + (mha.linear_o.weight, apb.linear_ada_out.weight, apb.linear_ada_out.bias)
    key = DR._key(params, cd, device)
    cache = blk.__dict__.get("_of3opt_dit_glue_w")
    if cache is None or cache["key"] != key:
        with torch.no_grad():
            w_qkvg = torch.cat([l.weight.detach() for l in lins], 0).to(device=device, dtype=cd).contiguous()                      # [(3|4)·HD, c]
            if any(l.bias is not None for l in lins):
                b_qkvg = torch.cat([(l.bias.detach() if l.bias is not None else torch.zeros(l.weight.shape[0], device=l.weight.device, dtype=l.weight.dtype))
                                    for l in lins], 0).to(device=device, dtype=cd).contiguous()
            else:
                b_qkvg = None
            w_o = mha.linear_o.weight.detach().to(device=device, dtype=cd).contiguous()
            b_o = mha.linear_o.bias.detach().to(device=device, dtype=cd).contiguous() if mha.linear_o.bias is not None else None
            w_ada = apb.linear_ada_out.weight.detach().to(device=device, dtype=cd).contiguous()                                   # [c, c_s]
            b_ada = apb.linear_ada_out.bias.detach().to(device=device, dtype=cd).contiguous()
        cache = {"key": key, "w_qkvg": w_qkvg, "b_qkvg": b_qkvg, "w_o": w_o, "b_o": b_o, "w_ada": w_ada, "b_ada": b_ada, "gated": getattr(mha, "linear_g", None) is not None}
        blk.__dict__["_of3opt_dit_glue_w"] = cache; DR.STATS["packed"] += 1
    return cache


def block_forward(blk, a, s, z, mask, mask_trans: bool = True):
    """The token DiT block on the fused schedule. a [.., N, c] (the residual stream), s [..', N, c_s], z the pair representation as the block
    receives it, mask [..', N] or None. Returns the new a (a's dtype and shape). Raises Refuse by name before any fused work."""
    import torch
    dk = _DK["mod"]
    apb = blk.attention_pair_bias; mha = apb.mha; tr = blk.conditioned_transition
    cd = DR.compute_dtype(a)
    c = int(a.shape[-1]); N = int(a.shape[-2]); H, D = int(mha.no_heads), int(mha.c_hidden); HD = H * D
    biases = apb._prep_bias(a=a, z=z, mask=mask)                                               # [mask bias, pair bias] as the module serves them (the pair cache applies here); a for shape only
    pair = biases[1]
    nb = 1
    for d_ in tuple(pair.shape[:-3]):
        nb *= int(d_)
    R = 1
    for d_ in tuple(a.shape[:-1]):
        R *= int(d_)
    S_ = R // N
    if pair.dim() < 3 or tuple(pair.shape[-3:]) != (H, N, N) or nb not in (1, S_):
        raise Refuse("bias_shape")                                                             # before any fused work
    x = DR.adaln_forward(apb.layer_norm_a, a, s)                                               # [.., N, c] in cd (s rows periodic); Refuse -> caller
    if x.dtype != cd or tuple(x.shape) != tuple(a.shape):
        raise Refuse("adaln_out:%s" % str(x.dtype).replace("torch.", ""))
    with torch.autocast("cuda", enabled=False):
        W = _weights(blk, cd, a.device)
        x2 = x.reshape(R, c)
        qkvg = torch.addmm(W["b_qkvg"], x2, W["w_qkvg"].t()) if W["b_qkvg"] is not None else x2 @ W["w_qkvg"].t()   # [R, (3|4)·HD]
        ncol = qkvg.shape[1]
        q5 = qkvg.view(S_, N, ncol // HD, H, D)
        g3 = qkvg.view(S_, N, ncol)
        pb = pair.reshape(nb, H, N, N)
        o = torch.empty((R, HD), device=a.device, dtype=cd); o3 = o.view(S_, N, HD)
        km2 = None; nm = 1
        if mask is not None:
            km2 = mask.reshape(-1, N)                                                            # rows of the token mask (nonzero = keep), read in place
            nm = km2.shape[0]
        scale = 1.0 / math.sqrt(D)
        served_by_core = False; bmode = None
        core = _apb_core() if nb == 1 else None
        if core is not None:
            pbc = pb[0]                                                                          # [H, N, N] as served
            if pbc.stride(-1) != 1 or pbc.dtype != cd:
                pbc = pbc.to(cd).contiguous(); cmode = "core:bias_copy"                          # ONE head-major relayout per call in the compute dtype
            else:
                cmode = "core:bias_direct"
            try:                                                                                 # all samples in one call, the bias read once for all S, operands in place
                core.pair_bias_attention(qkvg[:, 0:HD], qkvg[:, HD:2 * HD], qkvg[:, 2 * HD:3 * HD], pbc, key_mask=km2,
                                         gate=(qkvg[:, 3 * HD:4 * HD] if W["gated"] else None), num_samples=S_, num_heads=H, layout="rows",
                                         out=o, scale=scale, inf=float(getattr(apb, "inf", 1e9)))
                served_by_core = True; bmode = cmode; STATE["core_served"] += 1
            except core.Unsupported as e:                                                        # refused by name before any work -> dtk below, booked
                ev = getattr(e, "event", type(e).__name__)
                _count(STATE["core_refused"], ev)
                _log("the core's pair-bias attention refused (%s: %s); %s.flash_bias_attn serves this shape" % (ev, e, KERNEL), once_key=("core", ev))
        if not served_by_core:
            bias_h = None
            for i in range(S_):
                if bias_h is None or nb != 1:
                    bias_h = pb[0 if nb == 1 else i]
                    if bias_h.dtype != cd or bias_h.stride(-1) != 1:
                        bias_h = bias_h.to(cd).contiguous(); bmode = "dtk:bias_copy"            # one relayout per call when the served bias is not key-contiguous in the compute dtype
                    else:
                        bmode = "dtk:bias_direct"
                km = None if km2 is None else km2[0 if nm == 1 else i % nm]
                dk.flash_bias_attn(q5[i, :, 0].permute(1, 0, 2), q5[i, :, 1].permute(1, 0, 2), q5[i, :, 2].permute(1, 0, 2), bias=bias_h, key_mask=km,
                                   gate=(g3[i][:, 3 * HD:4 * HD] if W["gated"] else None), out=o3[i], scale=scale)
        u = torch.addmm(W["b_o"], o, W["w_o"].t()) if W["b_o"] is not None else o @ W["w_o"].t()  # [R, c]
        P = DR.lead_period(s.shape[:-1], a.shape[:-1])
        if P is None:
            s2 = s.expand(*a.shape[:-1], s.shape[-1]).reshape(-1, s.shape[-1]); P = R; smode = "expanded"
        else:
            s2 = s.reshape(-1, s.shape[-1]); smode = "direct" if P == R else "periodic"
        if s2.stride(-1) != 1:
            s2 = s2.contiguous()
        s2c = s2 if s2.dtype == cd else s2.to(cd)
        gg = torch.addmm(W["b_ada"], s2c, W["w_ada"].t())                                        # [P, c] adaLN-zero gate logits
        a2 = a.reshape(R, c)
        if a2.stride(-1) != 1:
            a2 = a2.contiguous()
        a_new = dk.gate_residual(u, gate=gg, gate_period=P, res=a2, out_dtype=a.dtype, sigmoid_gate=True).view(a.shape)
    out = DR.cond_transition_forward(tr, a_new, s, mask=(mask if mask_trans else None), residual=True)
    STATE["served"] += 1
    _count(STATE["modes"], "%s:%s:s=%s:S=%d" % (str(cd).replace("torch.", ""), bmode, smode, S_))
    if STATE["first"] is None:
        STATE["first"] = "a=%s:%s s=%s pair=%s:%s mask=%s cd=%s bias=%s" % (tuple(a.shape), str(a.dtype).replace("torch.", ""), tuple(s.shape), tuple(pair.shape),
                                                                            str(pair.dtype).replace("torch.", ""), None if mask is None else tuple(mask.shape), str(cd).replace("torch.", ""), bmode)
        _log("first served block call " + STATE["first"])
    return out


def _patch_block(mod) -> bool:
    Blk = mod.DiffusionTransformerBlock
    if getattr(Blk.forward, "_of3opt_dit_glue", False):
        return False
    orig = Blk.forward
    import inspect
    has_mask_trans = "_mask_trans" in inspect.signature(orig).parameters
    FLAGS = ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma", "use_high_precision_attention")

    def forward(self, a, s, z, mask=None, **kw):
        stock = lambda: orig(self, a=a, s=s, z=z, mask=mask, **kw)   # noqa: E731
        if STATE["state"] != "on" or getattr(self, "use_cross_attention", True):
            return stock()
        flags = {k: kw.get(k, False) for k in FLAGS}
        hp = bool(flags.get("use_high_precision_attention"))
        if hp:
            STATE["high_precision_asked"] += 1
        unknown = [k for k in kw if k not in FLAGS and k != "_mask_trans"]
        mask_trans = bool(kw.get("_mask_trans", True)) if has_mask_trans else True
        try:
            ok, why = (False, "kwargs:" + ",".join(sorted(unknown))) if unknown else plan(self, a, s, z, mask, flags, mask_trans=mask_trans)
        except Exception as e:  # noqa: BLE001
            from ..oom import is_oom
            if is_oom(e):
                raise
            ok, why = False, "plan_error:%s" % type(e).__name__
        if not ok:
            _count(STATE["fallback"], why)
            _log("block call a=%s -> the stock block (fallback:%s)" % (tuple(a.shape), why), once_key=("fb", why))
            return stock()
        try:
            out = block_forward(self, a, s, z, mask, mask_trans=mask_trans)
            if hp:
                STATE["high_precision_overridden"] += 1                                          # served on 16-bit operands (fp32 softmax statistics and accumulation): upstream's fp32-core word not honoured, by name
            return out
        except (Refuse, DR.Refuse) as e:                                                         # a by-name refusal before the fused work: the stock block, counted
            why = str(e) or "refused"
            _count(STATE["fallback"], why)
            _log("block call a=%s -> the stock block (fallback:%s)" % (tuple(a.shape), why), once_key=("fb", why))
            return stock()
        except Exception as e:  # noqa: BLE001                                                   # anything unexpected: named, counted, the stock block runs (the inputs are untouched)
            from ..oom import is_oom
            if is_oom(e):
                raise
            why = "error:%s" % type(e).__name__
            _count(STATE["fallback"], why); _count(STATE["errors"], why)
            _log("the schedule raised %r on a=%s -> the stock block for this call (counted as %s)" % (e, tuple(a.shape), why), once_key=("err", why))
            return stock()
    forward._of3opt_dit_glue = True; forward.__wrapped__ = orig
    Blk.forward = forward
    return True


def _rows_fallback(kind: str, why: str, shape) -> None:
    _count(STATE[kind + "_fallback"], why)
    _log("%s call a=%s -> the stock method (fallback:%s:%s)" % (kind, tuple(shape), kind, why), once_key=("rows", kind, why))


def _cond_forward_factory(orig):
    """ConditionedTransitionBlock.forward served by dit_rows.cond_transition_forward (the update; the caller adds the residual)."""
    def forward(self, a, s, mask=None, chunk_size=None, **kw):
        stock = lambda: orig(self, a, s, mask=mask, chunk_size=chunk_size, **kw)   # noqa: E731
        if STATE["rows_state"] != "on":
            return stock()
        import torch
        if kw:
            _rows_fallback("cond", "kwargs:" + ",".join(sorted(kw)), getattr(a, "shape", ())); return stock()
        if chunk_size is not None:
            _rows_fallback("cond", "chunked", a.shape); return stock()
        try:
            ok, why, mode = DR.plan_cond_transition(self, a, s, mask)
            if not ok:
                _rows_fallback("cond", why, a.shape); return stock()
            out = DR.cond_transition_forward(self, a, s, mask=mask, residual=False)
        except DR.Refuse as e:
            _rows_fallback("cond", str(e) or "refused", a.shape); return stock()
        except Exception as e:  # noqa: BLE001                                                   # anything unexpected: named, counted, the stock method runs (the inputs are untouched)
            from ..oom import is_oom
            if is_oom(e):
                raise
            why = "error:%s" % type(e).__name__
            _count(STATE["errors"], "cond:" + why); _rows_fallback("cond", why, a.shape); return stock()
        STATE["cond_served"] += 1
        _count(STATE["cond_modes"], "%s:%s%s" % (str(DR.compute_dtype(a)).replace("torch.", ""), mode, "" if mask is None else ":mask"))
        if STATE["cond_first"] is None:
            STATE["cond_first"] = "a=%s:%s s=%s mask=%s" % (tuple(a.shape), str(a.dtype).replace("torch.", ""), tuple(s.shape), None if mask is None else tuple(mask.shape))
            _log("first served conditioned-transition call " + STATE["cond_first"])
        return out
    forward._of3opt_dit_rows = True; forward.__wrapped__ = orig
    return forward


def _adaln_forward_factory(orig):
    """AdaLN.forward served by dit_rows.adaln_forward when the call's compute dtype is its input's (the output dtype then equals stock's)."""
    def forward(self, a, s, **kw):
        stock = lambda: orig(self, a, s, **kw)   # noqa: E731
        if STATE["rows_state"] != "on":
            return stock()
        import torch
        if kw:
            _rows_fallback("adaln", "kwargs:" + ",".join(sorted(kw)), getattr(a, "shape", ())); return stock()
        try:
            ok, why, _ = DR.plan_adaln(self, a, s)
            if not ok:
                _rows_fallback("adaln", why, a.shape); return stock()
            cd = DR.compute_dtype(a)
            if cd != a.dtype:                                                                    # a 16-bit autocast region over fp32 rows: stock returns its LayerNorm's dtype and the consumer may route on it
                _rows_fallback("adaln", "mixed_%s" % str(cd).replace("torch.", ""), a.shape); return stock()
            out = DR.adaln_forward(self, a, s)
        except DR.Refuse as e:
            _rows_fallback("adaln", str(e) or "refused", a.shape); return stock()
        except Exception as e:  # noqa: BLE001
            from ..oom import is_oom
            if is_oom(e):
                raise
            why = "error:%s" % type(e).__name__
            _count(STATE["errors"], "adaln:" + why); _rows_fallback("adaln", why, a.shape); return stock()
        STATE["adaln_served"] += 1
        return out
    forward._of3opt_dit_rows = True; forward.__wrapped__ = orig
    return forward


def _patch_rows(mod) -> str:
    """Serve ConditionedTransitionBlock.forward and AdaLN.forward class-wide (the classes as the block module imports them). Returns "" or the
    reason the rows are unavailable (the block schedule is unaffected)."""
    CTB = getattr(mod, "ConditionedTransitionBlock", None)
    if CTB is None:
        return "no ConditionedTransitionBlock in %s" % getattr(mod, "__name__", mod)
    tmod = sys.modules.get(getattr(CTB, "__module__", ""))
    Ada = getattr(tmod, "AdaLN", None) if tmod is not None else None
    if Ada is None:
        return "no AdaLN beside %s.ConditionedTransitionBlock" % getattr(tmod, "__name__", "?")
    if not getattr(CTB.forward, "_of3opt_dit_rows", False):
        CTB.forward = _cond_forward_factory(CTB.forward)
    if not getattr(Ada.forward, "_of3opt_dit_rows", False):
        Ada.forward = _adaln_forward_factory(Ada.forward)
    return ""


def census_line() -> str:
    core_c = None
    try:
        core_c = _CORE["mod"].census() if _CORE["mod"] is not None and hasattr(_CORE["mod"], "census") else None
    except Exception:  # noqa: BLE001
        pass
    return (f"{PREFIX} LEVER name=dit_glue state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" impl={STATE['impl']} min_tokens={STATE['min_tokens']} served={STATE['served']}"
            f" fallback={','.join('%s:%d' % kv for kv in sorted(STATE['fallback'].items())) or 'none'}"
            f" modes={','.join('%s:%d' % kv for kv in sorted(STATE['modes'].items())) or 'none'}"
            f" core={STATE['core'] or ('pending:' + STATE['core_pref'])} core_served={STATE['core_served']}"
            f" core_refused={','.join('%s:%d' % kv for kv in sorted(STATE['core_refused'].items())) or 'none'}"
            f" high_precision={HIGH_PRECISION} high_precision_asked={STATE['high_precision_asked']} high_precision_overridden={STATE['high_precision_overridden']}"
            + f" rows={STATE['rows'] if STATE['rows_state'] in ('on', 'off') else STATE['rows_state']} cond_served={STATE['cond_served']}"
            f" cond_fallback={','.join('%s:%d' % kv for kv in sorted(STATE['cond_fallback'].items())) or 'none'}"
            f" cond_modes={','.join('%s:%d' % kv for kv in sorted(STATE['cond_modes'].items())) or 'none'}"
            f" adaln_served={STATE['adaln_served']} adaln_fallback={','.join('%s:%d' % kv for kv in sorted(STATE['adaln_fallback'].items())) or 'none'}"
            + (f" core_census={core_c}" if core_c is not None else "") + (f" first={STATE['first']}" if STATE["first"] else "")
            + (f" cond_first={STATE['cond_first']}" if STATE["cond_first"] else ""))


def install(environ=None) -> dict:
    """Route the core kernel, patch DiffusionTransformerBlock.forward class-wide. Idempotent; raises by name when the core kernel is not
    importable."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not (M_DIT):
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_DIT=) before install")
    from opt_core.kernels import route
    route(KERNEL)
    import importlib
    dk = importlib.import_module(KERNEL)
    import inspect
    if "rowmask" not in inspect.signature(dk.gate_residual).parameters:
        raise RuntimeError(f"{PREFIX} {KERNEL} at {getattr(dk, '__file__', None)} carries no row-masked gate_residual (opt_core >= 0.5.18.20)")
    _DK["mod"] = dk; DR._DK["mod"] = dk
    STATE["min_tokens"] = int((environ.get(ENV_MIN) or "").strip() or MIN_DEFAULT)
    STATE["core_pref"] = (environ.get(ENV_CORE) or "auto").strip().lower()
    mod = importlib.import_module(M_DIT)
    _patch_block(mod)
    STATE["rows"] = ((environ.get(ENV_ROWS) if ENV_ROWS else None) or "all").strip().lower()
    if STATE["rows"] == "all":
        why = _patch_rows(mod)
        STATE["rows_state"] = "on" if not why else "unavailable:" + why
        if why:
            _log("the conditioned-transition / AdaLN rows are unavailable (%s): those calls keep the stock methods" % why)
    STATE["state"] = "on"; STATE["impl"] = getattr(dk, "__file__", None)
    STATE["installed"] = True
    _log(f"installed: DiffusionTransformerBlock.forward serves the token diffusion transformer's self-attention blocks on the fused schedule "
         f"(dit_rows AdaLN -> one q|k|v|g GEMM -> pair-bias attention [{'the core entry ' + CORE_MODULE + ' when carried, else ' if STATE['core_pref'] == 'auto' else ''}"
         f"{KERNEL}.flash_bias_attn] -> o GEMM -> gate+residual -> dit_rows conditioned transition) from {STATE['impl']}; min_tokens={STATE['min_tokens']}; "
         f"rows={STATE['rows']}" + (" (ConditionedTransitionBlock.forward and AdaLN.forward served class-wide by the dit_rows schedules: the atom transformer's "
                                    "and every other conditioned sub-layer outside the block schedule)" if STATE["rows_state"] == "on" else ""))
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
