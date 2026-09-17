"""odde_apb_bind -- OpenDDE's sampler attention sites served by the core's pair-bias-attention provider (opt_core.kernels.apb) BY WORD.

No kernel, cell table or row import lives here: every call asks the provider's ``select()`` for the row of its call class (compute capability,
dtype, cell geometry, sample count, token bucket, eager | graph) under the kit's WORD and serves it through the provider's own serving functions
(``pair_bias_attention`` / ``atom_attention``); the provider's census records each decision once per class (opt_core.cell_census).

Sites and switches (levers/SAMPLER; installed from odde_sampler.install)::

  ODDE_DIT_ATTN=<word>   the sampler's 24 DiffusionTransformer token AttentionPairBias modules (16 heads x 48, one [16,N,N] pair bias shared by the S
                         diffusion samples, sigmoid gate) -- ONE switch, the word picks the site:
      exact              lever dit_attn_exact: the module-global ``primitives._attention`` statement (stock's own decomposition kept: q pre-scaled by
                         stock, fp32 upcast, SDPA scale 1) served by the provider's EXACT tier = its bit-exact row where the provider vouches it on the
                         RUNNING stack (dit_exact: the sm_90 prebuilt for this torch/CUDA stack, self-checked bit-equal to torch SDPA at load), else the
                         provider names the stock op and the lever STEPS ASIDE BY NAME (nothing installed, the stock statement serves).
                         ``ODDE_DIT_ATTN_EXACT=1`` is a legacy spelling of the same word (read as an alias).
      fast | big       lever dit_attn_apb: the 24 modules' ``Attention.forward`` per instance + the fused token stack's attention (lever dit_fused)
                         served by the tier's fastest measured row per cell (fpf_apb / fpf_apb:fp16 / apb_attn / ... as the provider's table says per
                         card, dtype, sample count, token bucket and capture state).  ``apb`` is a legacy spelling of ``fast``.
      <row[:variant]>    one provider row pinned (ablation / A-B: fpf_apb, fpf_apb:fp16, apb_attn, sdpa:efficient, dit_exact, ...); a row the card or
                         stack cannot run is refused BY NAME at install.
      bf16               not this unit's word (levers/ACCEL dit_attn_bf16: the bf16 recast of the stock statement).
  ODDE_ATOM_ATTN=<word>  lever atom_attn_apb: the 3+3 atom-transformer attention modules (4 heads x 32, 32-query x 128-key local windows) per instance +
                         the fused atom stacks' attention (lever atom_fused): fast | big | <row> (fpf_atom | dtk_window | sdpa_gather); ``apb`` = fast.

Sixteen-bit activations (the fused token stack's precision word dit_lowp=fp16|bf16): the provider's DiT cells are keyed on the statement's dtype
(fp32; a bf16 family exists on some cards); a dtype the provider has no cell family for is keyed on the fp32 statement cell of the same geometry and
NAMED on the census (``cell_dtype=fp32(for fp16 operands)``) -- the fp32 cell's 16-bit-operand winner (fpf_apb:fp16) is the kernel 16-bit inputs
run.  Call forms a site's kernel rows do not serve (local attention at the token site, no bias, per-sample bias, other layouts) STEP ASIDE BY NAME to
the module's stock forward, counted per word.  A provider ``Refusal`` at call time (a bound the row names: int32
offsets, alignment) is counted by its word and THAT call takes the row the provider names instead (``fallback``), never silently.
Census per site: COUNTS[site] = word, calls (served through the provider), kernel_calls, stock_calls (the cell named a stock op), plan_calls (eager
calls that asked the graph-timed column inside the step graph's plan-for-capture window — `column()`), rows{row:variant: n},
cells{cell key: n}, asides / aside_<word>, refusals{kind: n}, installed_on, selection (the provider's describe() at install), notes.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional

import torch

from opt_core.kernels import apb as KA

NAME = "odde_apb_bind"
__version__ = "1"                     # the binding's version word (printed on the marker / LEVER lines)
SWITCH = {"dit": "ODDE_DIT_ATTN", "atom": "ODDE_ATOM_ATTN"}
LEGACY_EXACT_FLAG = "ODDE_DIT_ATTN_EXACT"                     # =1: a legacy spelling of ODDE_DIT_ATTN=exact (read as an alias; exported by no line)
LEGACY_WORDS = {"apb": "fast"}                                # a legacy spelling of the two site switches' fast word
FOREIGN_WORDS = {"dit": ("bf16",), "atom": ()}                # words of the same switch another unit owns (levers/ACCEL dit_attn_bf16)
CELL = {"dit": "dit_h16d48", "atom": "atom_h4d32w32x128", "exact": "dit_h16d48"}
GEOM = {"dit": (16, 48), "atom": (4, 32), "exact": (16, 48)}   # (heads, head dim) the cells are measured for; asserted on the modules at install
REP_SHAPE = (5, 800)                                          # (samples, tokens) of the install-time selection printed on the LEVER line (upstream's default --sample 5 at 800 tokens)
ASIDE_WORDS = {"dit": ("local_attention", "qkv_sample_dims", "no_bias", "per_sample_bias", "bias_shape"),
               "atom": ("call_form", "qkv_sample_dims", "per_sample_bias")}


def _fresh(site):
    return {"site": site, "word": None, "raw_word": None, "calls": 0, "kernel_calls": 0, "stock_calls": 0, "plan_calls": 0, "rows": {}, "cells": {}, "asides": 0,
            "refusals": {}, "installed_on": 0, "installed": False, "selection": None, "row": None, "notes": [], "error": None, "aside": None}


COUNTS: Dict[str, Dict[str, Any]] = {s: _fresh(s) for s in ("dit", "atom", "exact")}
LAUNCHES = {"n_apb": 0, "n_atom": 0}                           # kernel launches per site (ran-or-refuse counters: opendde_opt/ran.py)
_SEL: Dict[tuple, Any] = {}
_ABI = [None, False]                                          # (dit_exact_abi(), resolved?)
_ORIG_ATTENTION = None


class Aside(Exception):
    """A call form the site's rows do not serve: the caller runs the module's stock forward (counted by ``word``)."""

    def __init__(self, word):
        Exception.__init__(self, word)
        self.word = word


def _env(k, default=""):
    return os.environ.get(k, default).strip()


def word(site: str) -> Optional[str]:
    """The kit's word for `site` ('dit' | 'atom'): a provider tier word, a row[:variant] pin, or None (switch absent / another unit's word)."""
    raw = _env(SWITCH[site])
    lw = raw.lower()
    if site == "dit" and lw in ("", "0", "off") and _env(LEGACY_EXACT_FLAG, "0") not in ("", "0", "off"):
        COUNTS["exact"]["raw_word"] = f"{LEGACY_EXACT_FLAG}={_env(LEGACY_EXACT_FLAG)}"
        return "exact"
    if lw in ("", "0", "off") or lw in FOREIGN_WORDS.get(site, ()):
        return None
    if lw in LEGACY_WORDS:
        COUNTS[site]["raw_word"] = raw
        return LEGACY_WORDS[lw]
    return raw


def check_word(site: str, w: str) -> str:
    """Raise RuntimeError BY NAME for a word the provider does not know (neither a tier word nor a row)."""
    r, _v = KA.split_word(w)
    if r in KA.TIER_WORDS or r in KA.ROW_NAMES or r in KA.STOCK_ROWS:
        return w
    raise RuntimeError(f"{SWITCH.get(site, SWITCH['dit'])}={w}: not a provider word (tiers: {', '.join(KA.TIER_WORDS)}; rows: {', '.join(KA.ROW_NAMES)})")


def _is_exactish(w: str) -> bool:
    return KA.split_word(w)[0] in ("exact", "faithful", "dit_exact")


def _abi():
    if not _ABI[1]:
        _ABI[0], _ABI[1] = KA.dit_exact_abi(), True
    return _ABI[0]


def _cc():
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)


def selection(site: str, S: int, N: int, dtype, capture: bool, w: Optional[str] = None):
    """The provider's Selection for one call class of `site` (memoised per class).  `N`: tokens (dit / exact) or atoms (atom: the provider keys the
    windowed cells on tokens ~ atoms // 8, as its own atom_attention does).  A dtype without a measured family is keyed on the fp32 statement cell (named)."""
    w = w or COUNTS[site]["word"]
    dtw = KA.dtype_word(dtype)
    timing = column(w, capture)
    key = (site, w, int(S), int(N), dtw, bool(capture), timing)
    hit = _SEL.get(key)
    if hit is not None:
        return hit
    cc = _cc()
    cell = CELL[site]
    H, D = GEOM[site]
    cdt = dtw
    if dtw != "fp32" and not KA.cell_family(cc, dtw, cell, timing) and not KA.cell_family(cc, dtw, cell, "eager"):
        cdt = "fp32"
        note = f"cell_dtype=fp32(for {dtw} operands: no {KA.cc_word(cc)}|{dtw}|{cell} family measured)"
        if note not in COUNTS[site]["notes"]:
            COUNTS[site]["notes"].append(note)
    n = int(N) if site != "atom" else max(1, int(N) // 8)
    ex = _is_exactish(w)
    sel = KA.select(cc, cdt, cell, n, word=w, samples=int(S), timing=timing, capture=bool(capture), head_dim=D, heads=H,
                    stack=KA.stack_word() if ex else None, abi=_abi() if ex else None)
    _SEL[key] = sel
    st = COUNTS[site]
    st["cells"][str(sel.cell)] = st["cells"].get(str(sel.cell), 0)
    return sel


def _count_served(site: str, sel):
    st = COUNTS[site]
    arm = KA.arm_word(sel.row, sel.variant)
    st["calls"] += 1
    st["rows"][arm] = st["rows"].get(arm, 0) + 1
    st["cells"][str(sel.cell)] = st["cells"].get(str(sel.cell), 0) + 1
    if sel.row in KA.STOCK_ROWS:
        st["stock_calls"] += 1
    else:
        st["kernel_calls"] += 1
        if site in ("dit", "atom"):
            LAUNCHES["n_apb" if site == "dit" else "n_atom"] += 1


def _count_refusal(site: str, e):
    st = COUNTS[site]
    k = f"{getattr(e, 'row', None)}:{getattr(e, 'kind', e)}"[:96]
    st["refusals"][k] = st["refusals"].get(k, 0) + 1


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:                                                                  # pragma: no cover
        return False


PLAN_WORDS = ("big",)                                                                # the tier word(s) whose eager calls inside the step graph's plan-for-capture window ask the graph-timed
#   column: the big word ONLY — its fp32 unfused DiT cells name different eager / graph rows, so the eager reference the captured step is held against
#   must be served by the graph column's rows too. fast (its fused cells name the same row either way), exact, off and a row pin keep the literal capture
#   state.
STEPGRAPH_MODULE = "opendde_opt.stepgraph"                                             # the kit lever that owns the plan-for-capture window (read through sys.modules: no import either way)


def planned() -> bool:
    """True inside the sampler step graph's plan-for-capture window (opendde_opt.stepgraph.planning(): an admitted sampler call is running its
    denoiser steps — eager head, the capture's eager reference / warm-up / capture, the held replay's eager re-runs).  False when the lever's
    module is not loaded, outside a sampler call, or in a call the lever refused / stood aside from."""
    sg = sys.modules.get(STEPGRAPH_MODULE)
    try:
        return bool(sg is not None and sg.planning())
    except Exception:                                                                  # pragma: no cover  (a foreign module by that name)
        return False


def column(w: Optional[str], capture: bool) -> str:
    """The provider timing column one call asks: "graph" under capture (the provider's own rule: capture=True keys the graph-timed cells), and
    "graph" for EVERY call of the step graph's plan-for-capture window under the big word (PLAN_WORDS), so the eager reference the captured step is
    held against serves the rows the captured step serves (asked "eager" there, the big word's fp32 DiT cells would name a different row for the
    eager steps than for the captured one in the mid-size token buckets, the replay would not equal the reference and the graph would stand aside).
    "eager" otherwise — every call outside the window, and every fast / exact / row-pinned call not under capture, follows the literal capture state."""
    if capture:
        return "graph"
    if w in PLAN_WORDS and planned():
        return "graph"
    return "eager"


# ------------------------------------------------------------------------------------------------------------------ the token site (dit)
def dit_views(q, k, v, bias3, g=None, *, out_dtype=None, scale=None):
    """q/k/v(/g): [S, N, 16, 48] strided views (unit stride on the head dim); bias3: [16, N, >=N] fp32|bf16, unit last stride (one bias for the S
    samples).  Returns o [S, N, 16, 48] in out_dtype (default q.dtype) through the row the provider selects for this call class."""
    S, N, H, D = q.shape
    b4 = bias3[:, :N, :N][None]                                                         # [1, H, N, N] view: the provider's shared-bias form for every row
    cap = _capturing()
    sel = selection("dit", S, N, q.dtype, cap)
    if not cap and column(COUNTS["dit"]["word"], False) == "graph":
        COUNTS["dit"]["plan_calls"] += 1                                                # an EAGER call served from the graph column inside the step graph's window (census plan_calls=)
    sc = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    try:
        o, _ = KA.pair_bias_attention(q, k, v, b4, gate=g, selection=sel, layout="snhd", scale=sc, out_dtype=out_dtype or q.dtype)
    except KA.Refusal as e:                                                             # a bound the row names for THIS call: counted, and the row the provider names instead serves it
        _count_refusal("dit", e)
        fb = e.fallback or "sdpa"
        o, sel = KA.pair_bias_attention(q, k, v, b4, gate=g, word=fb, layout="snhd", scale=sc, out_dtype=out_dtype or q.dtype)
    _count_served("dit", sel)
    return o


def dit_packed(qkvg, bias3, S: int, N: int, *, out_dtype=None):
    """The fused token stack's entry (lever dit_fused): one q|k|v|g GEMM output [S*N, 4*768] -> gated attention output [S*N, 768] (views, no copies)."""
    H, D = GEOM["dit"]
    x = qkvg.view(S, N, 4, H, D)
    o = dit_views(x[:, :, 0], x[:, :, 1], x[:, :, 2], bias3, x[:, :, 3], out_dtype=out_dtype or qkvg.dtype)
    return o.reshape(S * N, H * D)


def _aside(site, aword, stock, att, q_x, kv_x, kw):
    st = COUNTS[site]
    st["asides"] += 1
    st["aside_" + aword] = st.get("aside_" + aword, 0) + 1
    return stock(att, q_x, kv_x, **kw)


def asides(site: str) -> dict:
    st = COUNTS[site]
    return {w: int(st.get("aside_" + w) or 0) for w in ASIDE_WORDS.get(site, ()) if st.get("aside_" + w)}


def _make_dit_forward(att):
    H = att.num_heads
    st = COUNTS["dit"]
    stock = type(att).forward                                                           # the module's stock forward (the class method this install shadows) serves the asides

    def forward(q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
        kw = dict(attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias, n_queries=n_queries, n_keys=n_keys, inf=inf, inplace_safe=inplace_safe, chunk_size=chunk_size)
        if n_queries or trunked_attn_bias is not None:                                  # a local-attention call form: not this site's rows -- stock serves it, by name
            return _aside("dit", "local_attention", stock, att, q_x, kv_x, kw)
        if attn_bias is None:
            return _aside("dit", "no_bias", stock, att, q_x, kv_x, kw)
        C = att.linear_q.weight.shape[0]; D = C // H
        lead = q_x.shape[:-2]; N = q_x.shape[-2]
        S = 1
        for d_ in q_x.shape[:-2]:
            S *= int(d_)
        Skv = 1
        for d_ in kv_x.shape[:-2]:
            Skv *= int(d_)
        if Skv != S or kv_x.shape[-2] != N:                                              # q / kv sample dims differ (cross-attention forms): stock, by name
            return _aside("dit", "qkv_sample_dims", stock, att, q_x, kv_x, kw)
        b = attn_bias
        while b.dim() > 3 and b.shape[0] == 1:
            b = b[0]
        if b.dim() == 4 and b.shape[0] == S and b.shape[1] == 1 and S != H:               # [S,1,N,N]: a per-sample mask-only bias -- not this site's contract: stock, by name
            return _aside("dit", "per_sample_bias", stock, att, q_x, kv_x, kw)
        if b.dim() != 3 or b.shape[0] != H or b.shape[-1] < N or b.shape[-2] < N:
            return _aside("dit", "bias_shape", stock, att, q_x, kv_x, kw)
        q = att.linear_q(q_x); k = att.linear_k(kv_x); v = att.linear_v(kv_x)
        g = att.linear_g(q_x) if att.gating else None
        q4, k4, v4 = (t.reshape(-1, N, H, D) for t in (q, k, v))
        g4 = g.reshape(-1, N, H, D) if g is not None else None
        if b.stride(-1) != 1:
            b = b.contiguous()
        o = dit_views(q4, k4, v4, b, g4, out_dtype=q.dtype, scale=1.0 / math.sqrt(D))
        return att.linear_o(o.reshape(*lead, N, C))

    forward.__wrapped__ = type(att).forward; forward._odde_apb_site = "dit"
    st["installed"] = True
    return forward


def _diffusion_module(model):
    from opendde.model.modules import diffusion as Dm
    dms = [m for m in model.modules() if isinstance(m, Dm.DiffusionModule)]
    if len(dms) != 1:
        raise RuntimeError(f"expected one DiffusionModule in the model, found {len(dms)}")
    return dms[0]


def _install_selection(site: str, w: str):
    """Resolve the word once at install (the representative class printed on the LEVER line); a ROW pin the card / stack cannot run raises BY NAME."""
    st = COUNTS[site]
    st["word"] = check_word(site, w)
    try:
        sel = selection(site, REP_SHAPE[0], REP_SHAPE[1] if site != "atom" else REP_SHAPE[1] * 8, torch.float32, False, w)
    except KA.Refusal as e:                                                             # a pinned row this card / stack cannot run (tier words never raise here: they name the stock op)
        raise RuntimeError(f"{SWITCH['dit' if site == 'exact' else site]}={w}: the provider refuses this row here ({e})") from e
    st["selection"] = KA.describe(sel); st["row"] = KA.arm_word(sel.row, sel.variant)
    return sel


def install_dit(model) -> dict:
    """Lever dit_attn_apb: put the provider-served forward on the 24 token-attention modules (word fast | big | <row>)."""
    st = COUNTS["dit"]
    try:
        w = word("dit")
        if w is None or (_is_exactish(w) and KA.split_word(w)[0] != "dit_exact"):
            raise RuntimeError(f"install_dit called with {SWITCH['dit']}={_env(SWITCH['dit'])!r} (a fast | big | <row> word installs this site; exact is install_exact's)")
        _install_selection("dit", w)
        from opendde.model.modules import primitives as P
        dm = _diffusion_module(model)
        mods = [blk.attention_pair_bias.attention for blk in dm.diffusion_transformer.blocks]
        if len(mods) != 24 or any(not isinstance(m, P.Attention) for m in mods):
            raise RuntimeError(f"expected 24 primitives.Attention token modules under diffusion_transformer.blocks, found {len(mods)}")
        H, D = GEOM["dit"]
        for att in mods:
            if att.num_heads != H or att.c_hidden != D:                                # primitives.Attention: c_hidden IS the per-head width
                raise RuntimeError(f"token attention geometry heads={att.num_heads} c_hidden={att.c_hidden} (cells measured for {H} x {D})")
            att.forward = _make_dit_forward(att)
        st["installed_on"] = len(mods)
        return dict(st)
    except Exception as e:
        st["error"] = repr(e)
        raise RuntimeError(f"dit_attn_apb: {e}") from e


# ------------------------------------------------------------------------------------------------------------------ the atom site
def atom_views(q, k, v, bias, g=None, *, n_queries=32, n_keys=128, out_dtype=None, scale=None):
    """q/k/v(/g): [S, NA, 4, 32] strided views; bias: [4, n_windows, n_queries, n_keys] (leading 1s allowed).  Returns o [S, NA, 4, 32]."""
    S, NA, H, D = q.shape
    b = bias
    while b.dim() > 4 and b.shape[0] == 1:
        b = b[0]
    cap = _capturing()
    sel = selection("atom", S, NA, q.dtype, cap)
    if not cap and column(COUNTS["atom"]["word"], False) == "graph":
        COUNTS["atom"]["plan_calls"] += 1
    sc = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    try:
        o, _ = KA.atom_attention(q, k, v, b, g, selection=sel, n_queries=n_queries, n_keys=n_keys, scale=sc, out_dtype=out_dtype or q.dtype)
    except KA.Refusal as e:
        _count_refusal("atom", e)
        o, sel = KA.atom_attention(q, k, v, b, g, word=e.fallback or "sdpa_gather", n_queries=n_queries, n_keys=n_keys, scale=sc, out_dtype=out_dtype or q.dtype)
    _count_served("atom", sel)
    return o


def _local_only(P) -> bool:
    """OpenDDE's primitives.Attention.forward routes every n_queries/n_keys call to primitives._local_attention (no method attribute): true when the
    class source carries that single local statement (the pinned 1.1.1 text)."""
    import inspect
    try:
        src = inspect.getsource(P.Attention.forward)
    except Exception:                                                                  # pragma: no cover
        return False
    return "_local_attention(" in src and "local_attention_method" not in src and hasattr(P, "_local_attention")


def _make_atom_forward(att):
    H = att.num_heads
    stock = type(att).forward

    def forward(q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
        kw = dict(attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias, n_queries=n_queries, n_keys=n_keys, inf=inf, inplace_safe=inplace_safe, chunk_size=chunk_size)
        if not (n_queries and n_keys) or trunked_attn_bias is None or attn_bias is not None:   # not the windowed local-attention form: stock, by name
            return _aside("atom", "call_form", stock, att, q_x, kv_x, kw)
        C = att.linear_q.weight.shape[0]; D = C // H
        lead = q_x.shape[:-2]; N = q_x.shape[-2]
        S = 1
        for d_ in q_x.shape[:-2]:
            S *= int(d_)
        Skv = 1
        for d_ in kv_x.shape[:-2]:
            Skv *= int(d_)
        if Skv != S or kv_x.shape[-2] != N:
            return _aside("atom", "qkv_sample_dims", stock, att, q_x, kv_x, kw)
        b = trunked_attn_bias
        while b.dim() > 4:
            if b.shape[0] != 1:                                                         # a per-sample local pair bias (the model's is sample-invariant): stock, by name
                return _aside("atom", "per_sample_bias", stock, att, q_x, kv_x, kw)
            b = b[0]
        q = att.linear_q(q_x); k = att.linear_k(kv_x); v = att.linear_v(kv_x)
        g = att.linear_g(q_x) if att.gating else None
        q4, k4, v4 = (t.reshape(-1, N, H, D) for t in (q, k, v))
        g4 = g.reshape(-1, N, H, D) if g is not None else None
        if b.stride(-1) != 1:
            b = b.contiguous()
        o = atom_views(q4, k4, v4, b, g4, n_queries=n_queries, n_keys=n_keys, out_dtype=q.dtype, scale=1.0 / math.sqrt(D))
        return att.linear_o(o.reshape(*lead, N, C))

    forward.__wrapped__ = type(att).forward; forward._odde_apb_site = "atom"
    COUNTS["atom"]["installed"] = True
    return forward


def install_atom(model) -> dict:
    """Lever atom_attn_apb: put the provider-served forward on the 6 atom-transformer attention modules (word fast | big | <row>)."""
    st = COUNTS["atom"]
    try:
        w = word("atom")
        if w is None:
            raise RuntimeError(f"install_atom called with {SWITCH['atom']} unset")
        _install_selection("atom", w)
        from opendde.model.modules import primitives as P, transformer as T
        dm = _diffusion_module(model)
        mods = []
        for part in (dm.atom_attention_encoder, dm.atom_attention_decoder):
            for m in part.modules():
                if isinstance(m, T.AtomTransformer):
                    mods += [blk.attention_pair_bias.attention for blk in m.diffusion_transformer.blocks]
        if len(mods) != 6 or any(not isinstance(m, P.Attention) for m in mods):
            raise RuntimeError(f"expected 6 primitives.Attention modules under the diffusion module's atom encoder/decoder AtomTransformers, found {len(mods)}")
        H, D = GEOM["atom"]
        for att in mods:
            meth = getattr(att, "local_attention_method", "local_cross_attention" if _local_only(P) else None)
            if meth != "local_cross_attention":
                raise RuntimeError(f"local_attention_method={meth!r} (the rows implement 'local_cross_attention')")
            if att.num_heads != H or att.c_hidden != D:
                raise RuntimeError(f"atom attention geometry heads={att.num_heads} c_hidden={att.c_hidden} (cells measured for {H} x {D})")
            att.forward = _make_atom_forward(att)
        st["installed_on"] = len(mods)
        return dict(st)
    except Exception as e:
        st["error"] = repr(e)
        raise RuntimeError(f"atom_attn_apb: {e}") from e


# ------------------------------------------------------------------------------------------------------------------ the exact route (dit, word exact)
def _route_reason(q, k, v, b) -> Optional[str]:
    """None = ask the provider; else the word for the stock statement (decided from ranks / dtypes / shapes only; layout bounds are the row's own, named by its Refusal)."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or b.dim() != 4:
        return "rank"
    if q.shape[-1] != GEOM["exact"][1] or q.shape[1] != GEOM["exact"][0]:
        return "geometry"
    if not (q.is_cuda and q.dtype == torch.float32 and k.dtype == torch.float32 and v.dtype == torch.float32 and b.dtype == torch.float32):
        return "dtype"
    B, H, N, D = q.shape
    if tuple(k.shape) != (B, H, N, D) or tuple(v.shape) != (B, H, N, D):
        return "kv_shape"
    if b.shape[0] not in (1, B) or tuple(b.shape[1:]) != (H, N, N):
        return "bias_shape"
    return None


def _lead_view(q, k, v, b):                                                            # [1,...,1,B,H,N,D] -> [B,H,N,D] views (OpenDDE's sampler batch dim); None = not viewable
    if q.dim() <= 4:
        return q, k, v, b, ()
    extra = q.dim() - 4
    if any(int(x) != 1 for x in q.shape[:extra]) or k.dim() != q.dim() or v.dim() != q.dim() or any(int(x) != 1 for x in k.shape[:extra]) or any(int(x) != 1 for x in v.shape[:extra]):
        return q, k, v, b, None
    lead = tuple(q.shape[:extra])
    q4, k4, v4 = q.reshape(q.shape[extra:]), k.reshape(k.shape[extra:]), v.reshape(v.shape[extra:])
    b4 = b
    while b4.dim() > 4 and int(b4.shape[0]) == 1:
        b4 = b4[0]
    if b4.dim() != 4:
        return q, k, v, b, None
    return q4, k4, v4, b4, lead


def _count_route(r: str):
    st = COUNTS["exact"]
    st["routes"] = st.get("routes") or {}
    st["routes"][r] = st["routes"].get(r, 0) + 1


def _exact_probe(sel) -> Optional[str]:
    """Serve two small deterministic cases through the selected row and compare with the stock statement in THIS process (torch.equal); None = bit-equal,
    else the word.  (The provider self-checks its prebuilt at load as well; a Refusal there is returned as its word.)"""
    import torch.nn.functional as F
    dev = torch.device("cuda")
    for i, (B, N) in enumerate(((5, 200), (2, 100))):
        gen = torch.Generator(device="cpu"); gen.manual_seed(4800 + i)
        H, D = GEOM["exact"]

        def lin():
            return torch.randn(B, N, H * D, generator=gen, dtype=torch.float32).to(dev).view(B, N, H, D).transpose(1, 2)
        q = lin() / math.sqrt(D); k = lin(); v = lin()
        b = torch.randn(1, H, N, N, generator=gen, dtype=torch.float32).to(dev)
        try:
            o, _ = KA.pair_bias_attention(q, k, v, b, selection=sel, layout="shnd", scale=1.0, out_dtype=torch.float32)
        except KA.Refusal as e:
            return f"provider_refusal:{e.kind}"[:120]
        ref = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=b, scale=1.0)
        if o.shape != ref.shape or not torch.equal(o, ref):
            return "probe_not_bit_equal_vs_sdpa"
    return None


def install_exact() -> str:
    """Lever dit_attn_exact (ODDE_DIT_ATTN=exact): bind primitives._attention when the provider's exact tier serves a kernel row on this stack; else step
    aside BY NAME (COUNTS['exact']['aside'] = the word; nothing installed, the stock statement serves).  Returns the marker line."""
    global _ORIG_ATTENTION
    st = COUNTS["exact"]
    w = word("dit")
    if w is None or not _is_exactish(w):
        return f"DITATTN:inactive({SWITCH['dit']}={_env(SWITCH['dit'])!r})"
    if st["installed"] or st.get("aside"):
        return st.get("marker")
    if not torch.cuda.is_available():
        st["aside"] = "no_cuda_device"
    else:
        sel = _install_selection("exact", w)
        if sel.row in KA.STOCK_ROWS:                                                    # the provider's exact tier names the stock op on this card / stack: the statement stays stock's
            st["aside"] = f"provider_exact_names_stock:{KA.arm_word(sel.row, sel.variant)}@sm{'%d%d' % _cc()}"
        else:
            why = _exact_probe(sel)
            if why is not None:                                                         # exact never ships unequal bytes: the stock statement serves, by name
                st["aside"] = why
    if st.get("aside"):
        st["marker"] = f"DITATTN:aside({st['aside']}; the stock fp32 SDPA statement serves -- provider: {st.get('selection')})"
        return st["marker"]
    import opendde.model.modules.primitives as PR
    _ORIG_ATTENTION = PR._attention
    _orig = _ORIG_ATTENTION

    def _attention(q, k, v, attn_bias=None, use_efficient_implementation=True, inplace_safe=False):
        if use_efficient_implementation and attn_bias is not None and q.device.type != "mps":
            input_dtype = q.dtype                                                       # OpenDDE's statement (primitives.py:283-309) upcasts q, k AND v and casts the output back
            q32 = q.to(dtype=torch.float32); k32 = k.to(dtype=torch.float32); v32 = v.to(dtype=torch.float32); b32 = attn_bias.to(dtype=torch.float32)
            q4, k4, v4, b4, lead = _lead_view(q32, k32, v32, b32)
            why = "lead_dims" if lead is None else _route_reason(q4, k4, v4, b4)
            if why is None:
                S, N = int(q4.shape[0]), int(q4.shape[2])
                sel = selection("exact", S, N, torch.float32, _capturing())
                if sel.row in KA.STOCK_ROWS:                                            # this class's exact cell names the stock op: the statement below IS it
                    _count_route(f"cell_stock:{KA.arm_word(sel.row, sel.variant)}"); st["stock_calls"] += 1; st["calls"] += 1
                else:
                    try:
                        o, _s = KA.pair_bias_attention(q4, k4, v4, b4, selection=sel, layout="shnd", scale=1.0, out_dtype=torch.float32)
                        _count_served("exact", sel); _count_route("kernel")
                        if lead:
                            o = o.reshape(tuple(lead) + tuple(o.shape))
                        return o.to(dtype=input_dtype)
                    except KA.Refusal as e:                                             # the row's own bound for this call (alignment / size): the stock statement, by name
                        _count_refusal("exact", e); _count_route(f"refusal:{e.kind}"[:64]); st["calls"] += 1; st["stock_calls"] += 1
            else:
                _count_route(why); st["calls"] += 1; st["stock_calls"] += 1
        else:
            _count_route("not_sdpa" if not use_efficient_implementation else ("mps" if q.device.type == "mps" else "no_bias")); st["calls"] += 1; st["stock_calls"] += 1
        return _orig(q, k, v, attn_bias=attn_bias, use_efficient_implementation=use_efficient_implementation, inplace_safe=inplace_safe)

    _attention._dit_attn_exact = True
    PR._attention = _attention
    st["installed"] = True; st["installed_on"] = 1
    st["marker"] = f"DITATTN:on({NAME} {__version__}; provider {st['selection']}; probe 2 cases bit-equal vs torch SDPA in this process)"
    return st["marker"]


def report() -> dict:
    return {k: dict(v) for k, v in COUNTS.items()}


def reset():
    """Tests: forget every count / selection (the environment is re-read on the next install)."""
    global _ORIG_ATTENTION
    for s in COUNTS:
        COUNTS[s].clear(); COUNTS[s].update(_fresh(s))
    _SEL.clear(); LAUNCHES.update(n_apb=0, n_atom=0)
    if _ORIG_ATTENTION is not None:
        try:
            import opendde.model.modules.primitives as PR
            PR._attention = _ORIG_ATTENTION
        except Exception:                                                              # pragma: no cover
            pass
        _ORIG_ATTENTION = None
