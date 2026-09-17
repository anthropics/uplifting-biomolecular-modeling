"""The sampler / pairformer attention-with-pair-bias levers bound through the shared core's provider BY TIER WORD, on every card.

One binding for the whole family, on every card.  Every call goes to ``opt_core.kernels.apb`` with the mode's tier word
(``exact`` under --mode exact, ``fast`` under fast, ``big`` under big); the provider's measured cell table (APB_CELLS.json) names the row per
call class (card, dtype, geometry cell, samples, token bucket, eager | graph capture) and serves it.  Nothing here pins a row, a tile or a size
floor: a decision the table gets wrong is fixed in the table (common/opt_core), not here.

  dit_attn        the 24 DiffusionTransformer blocks' token attention (q,k,v [S,N,16,48], pair bias [16,N,N] shared by the S samples, gate fused):
                  ``pair_bias_attention(word=<tier>, cell="dit_h16d48")`` per call.  The provider's stock-family rows (``sdpa:*``) are served
                  through the same face when the table names them the cell's winner (small eager buckets) -- counted ``served=sdpa:auto:<n>``,
                  a measured decision, not a fallback.  A ``Refusal`` (a geometry / offset range no admitted row serves) takes the module's own
                  statement BY NAME, counted ``aside=<kind>:<n>`` on the LEVER line.
  dit_attn_fp16   rides dit_attn: ON = the fp16-operand precision arms (``<row>:fp16``) are admissible to the tier word (the table serves them
                  where they are the measured winner); ablated (MODEL_OPT_LEVERS_OFF=dit_attn_fp16) = a call class whose winner is an fp16 arm is
                  re-decided with the provider's own ``prefer=(<row>,)`` rule (the row's plain arm where measured, else the cell's stock arm).
  atom_attn       the atom encoder + decoder local attention (q,k,v [S,N_atom,4,32], 32x128 windows): ``atom_attention(word=<tier>)``.
  pf_attn         the Pairformer / confidence-head AttentionPairBias (has_s=False; bf16 autocast): producer ``pair_bias_planes(word=<tier>)``
                  (LayerNorm(z) + Linear c_z->16 head-major planes) + core ``pair_bias_attention(word=<tier>, cell="pf_h16d24")`` on the stock
                  projections.  The dtype gate (dtype_gate.py) stays in front: outside bf16 autocast the call steps aside by name.
  dit_attn_exact  (exact rows of cc 9.0) ``primitives._attention`` routed per call by ``select(word="exact", stack=<this stack>, abi=<torch/CUDA
                  key>)``: the provider's ``dit_exact`` row (its carried prebuilt sm_90 kernel, bit-identical to the memory-efficient SDPA kernel,
                  vouched per stack in the table) serves the classes the table vouches on THIS stack; a class the exact word resolves to a
                  stock-family arm keeps the statement itself BY NAME (counted ``exact_word:<arm>``) -- exact bytes never depend on a kernel the
                  table has not vouched here.  ``apply()`` / ``report()`` keep the hook contract src/ptx_trunk2_levers.apply_from_env calls.

atom_attn_exact binds the same face by the word ``exact`` from ``protenix_opt.apb_atom_exact`` (the runner-seam package protocol of
sampler_levers): row ``atom_exact`` where the provider's cells vouch it on the stack / card, the stock statement BY NAME elsewhere.

pf_attn timing column: the provider keys its cells by timing (eager | graph; capture=True reads the graph column) and the two columns may
name different rows for pf_h16d24.  The 48 Pairformer
sites sit inside the trunk stack graph's region: a call fpf_stackgraph is PLANNING to capture (its eager reference and warm-up of a first-sighted
signature) selects the graph column exactly like the captured call (``_timing``), so the graph's bitwise oracle and the graph serve one row;
``capture=`` stays the literal capture state.  fast | big only (pf_attn is not an exact-mode lever; the exact word would key on the capture state alone).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

NAME = "apb_core"
FACE = "opt_core.kernels.apb"
LEVERS = ("dit_attn", "atom_attn", "pf_attn")                   # the tolerance-tier levers installed on the runner seam (dit_attn_fp16 rides dit_attn)
EXACT_LEVER = "dit_attn_exact"
TIER_WORDS = {"exact": "exact", "fast": "fast", "big": "big"}   # mode -> the provider's tier word (kernels.apb.TIER_WORDS carries all three)
CELLS = {"dit_attn": "dit_h16d48", "atom_attn": "atom_h4d32w32x128", "pf_attn": "pf_h16d24"}   # the provider's geometry words for the three sites (kernels.apb.CELL_WORDS)
DIT_HEADS, DIT_HEAD_DIM = 16, 48                                # DiffusionTransformer token attention (protenix v2: 24 blocks, c_token 768)
ATOM_HEADS, ATOM_HEAD_DIM, ATOM_NQ, ATOM_NK = 4, 32, 32, 128    # AtomTransformer local attention (encoder 3 + decoder 3 blocks)
ATOM_METHOD = "local_cross_attention"                           # primitives.Attention.local_attention_method of those modules (n_queries x n_keys windows over the atoms)
PF_HEADS = 16                                                   # Pairformer AttentionPairBias (c_s 384 -> 16 x 24)
FACE_NAMES = ("select", "pair_bias_attention", "atom_attention", "pair_bias_planes", "carried_module", "stack_word", "dit_exact_abi",
              "arm_word", "Refusal", "STOCK_ROWS", "TIER_WORDS")   # the provider surface this binding calls, checked BY NAME at install (an older core refuses by name, never a silent stock run)
MARK = {"dit_attn": "DIT_ATTN:", "atom_attn": "ATOM_ATTN:", "pf_attn": "PF_ATTN:", "dit_attn_exact": "DITATTN:", "atom_attn_exact": "ATOMATTNEXACT:"}


def _fresh_state() -> Dict[str, Dict[str, Any]]:
    return {
        "dit_attn": {"installed_on": 0, "calls": 0, "cell": None, "cell_key": None, "named": None, "error": None, "precision": None,
                     "served": {}, "aside": {}, "left_alone": []},
        "dit_attn_fp16": {"on": False, "error": None, "engaged_calls": 0},
        "atom_attn": {"installed_on": 0, "calls": 0, "cell": None, "cell_key": None, "named": None, "error": None, "served": {}, "aside": {}},
        "pf_attn": {"installed_on": 0, "calls": 0, "cell": None, "cell_key": None, "named": None, "error": None, "served": {}, "aside": {},
                    "producer": {}, "skipped_has_s": 0},
        "dit_attn_exact": {"installed": False, "calls": 0, "routes": {}, "marker": None, "loadcheck": None, "row": None, "stack": None, "abi": None,
                           "error": None},
    }


STATE: Dict[str, Dict[str, Any]] = _fresh_state()
_SEL: Dict[tuple, Any] = {}                                     # (site, cc, dtype, N, S, capture, word, fp16_ok) -> Selection | Refusal: one provider decision per call class
_DX: Dict[str, Any] = {"mod": None, "orig": None, "pkg": None}   # dit_attn_exact: the loaded prebuilt module, the original statement, the provider's carried package


def _count(d: dict, key: str, n: int = 1) -> None:
    d[key] = int(d.get(key, 0)) + n


# ----------------------------------------------------------------------------------------------------------------- the provider, by name
def face():
    """``opt_core.kernels.apb`` with every name this binding calls present, else RuntimeError BY NAME (the lever line says unavailable)."""
    import opt_core
    from opt_core.kernels import apb as A
    missing = [n for n in FACE_NAMES if not hasattr(A, n)]
    if missing:
        raise RuntimeError(f"{FACE} of opt_core {getattr(opt_core, '__version__', '?')} lacks {missing}: this kit's attention levers need a newer shared core")
    lost = [w for w in TIER_WORDS.values() if w not in tuple(A.TIER_WORDS)]
    if lost:
        raise RuntimeError(f"{FACE} of opt_core {getattr(opt_core, '__version__', '?')} does not carry the tier words {lost}")
    return A


def core_version() -> str:
    try:
        import opt_core
        return str(getattr(opt_core, "__version__", "?"))
    except Exception:  # noqa: BLE001
        return "?"


def tier_word(mode: Optional[str] = None) -> str:
    """The provider tier word of the active mode: ``mode`` given, else the activation record of this process (stack.status), else the env
    route's selection variable PROTENIX_OPT.  A mode without a word raises by name (never a guessed word)."""
    m = (mode or "").strip().lower()
    if not m:
        try:
            from . import stack as _stack
            m = str((_stack.status() or {}).get("mode") or "").strip().lower()
        except Exception:  # noqa: BLE001
            m = ""
    if not m:
        m = (os.environ.get("PROTENIX_OPT") or "").strip().lower()
    if m not in TIER_WORDS:
        raise RuntimeError(f"{NAME}: no provider tier word for mode {m or '<unknown>'!r} (expected one of {sorted(TIER_WORDS)})")
    return TIER_WORDS[m]


def _cc(t) -> tuple:
    import torch
    return tuple(torch.cuda.get_device_capability(t.device)) if t.is_cuda else (0, 0)


def _cc_word(cc: tuple) -> str:
    return "%d.%d" % (int(cc[0]), int(cc[1]))


def _capturing(t) -> bool:
    import torch
    return bool(t.is_cuda and torch.cuda.is_current_stream_capturing())


def _stackgraph_planning() -> bool:
    """True while the trunk stack graph (forward/flashpairformer/src/fpf_stackgraph) runs the eager reference / warm-up / capture of a signature it is
    about to capture (its plan-for-capture window); False when the module is not loaded (big: no stack graph) or predates the window."""
    for name in ("fpf_stackgraph.stackgraph", "fpf_stackgraph"):
        f = getattr(sys.modules.get(name), "planning", None)
        if f is not None:
            try:
                return bool(f())
            except Exception:  # noqa: BLE001 -- a foreign / partial module: no window
                return False
    return False


def _timing(word: str, capture: bool) -> str:
    """The provider timing column of a pf_attn call: 'graph' inside a capture and, under a tier word (fast | big), for a call the trunk stack graph
    plans to capture -- so its eager reference and the captured replay serve ONE row; 'eager' otherwise (the exact word: the capture state alone)."""
    return "graph" if (capture or (word != "exact" and _stackgraph_planning())) else "eager"


def _select(A, site: str, t, *, N: int, S: int, heads: int, head_dim: int, word: str, capture: bool, fp16_ok: bool = True,
            stack: Optional[str] = None, abi: Optional[str] = None, cell: Optional[str] = None, c_z: Optional[int] = None, timing: Optional[str] = None):
    """The provider's decision for one call class, memoised (the provider records its census once per class; the decision is pure given the
    table).  ``site`` atom_attn measures tokens as the face does for the windowed cell (atoms // 8)."""
    cc = _cc(t)
    timing = timing or ("graph" if capture else "eager")               # the provider's column: capture=True reads the graph column on its own; a PLANNED eager call (pf_attn) names it
    key = (site, cc, str(t.dtype), int(N), int(S), bool(capture), word, bool(fp16_ok), stack, abi) + ((timing,) if timing == "graph" and not capture else ())
    hit = _SEL.get(key)
    if hit is not None:
        if isinstance(hit, Exception):
            raise hit
        return hit
    cell = cell or CELLS[site if site in CELLS else "dit_attn"]
    n_tokens = max(1, int(N) // 8) if site == "atom_attn" else int(N)
    kw = dict(word=word, samples=int(S), capture=bool(capture), heads=heads, head_dim=head_dim, stack=stack, abi=abi, timing=timing)
    if c_z is not None:
        kw = dict(word=word, samples=int(S), capture=bool(capture), heads=heads, c_z=c_z, stack=stack, timing=timing)
    try:
        sel = A.select(cc, t.dtype, cell, n_tokens, **kw)
        if site == "dit_attn" and not fp16_ok and (sel.variant or "") == "fp16":   # dit_attn_fp16 ablated: the precision arm is inadmissible -> the provider's prefer rule on the same row
            sel = A.select(cc, t.dtype, cell, n_tokens, prefer=(sel.row,), **kw)
    except A.Refusal as e:
        _SEL[key] = e
        raise
    _SEL[key] = sel
    return sel


def selection_summary(site: str) -> Dict[str, int]:
    """{arm: number of call classes decided for it} over this process's memo for ``site`` (refusals as ``refused:<kind>``)."""
    out: Dict[str, int] = {}
    A = None
    for key, v in _SEL.items():
        if key[0] != site:
            continue
        if isinstance(v, Exception):
            _count(out, "refused:" + str(getattr(v, "kind", "error")))
            continue
        if A is None:
            A = face()
        _count(out, A.arm_word(v.row, v.variant))
    return out


# ----------------------------------------------------------------------------------------------------------------- model matching
def _diffusion_module(model):
    """The model's DiffusionModule (protenix.model.modules.diffusion.DiffusionModule): the module itself or the one instance under ``model``."""
    from protenix.model.modules import diffusion as Dm
    if isinstance(model, Dm.DiffusionModule):
        return model
    found = [m for m in model.modules() if isinstance(m, Dm.DiffusionModule)]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one DiffusionModule under the model, found {len(found)}")
    return found[0]


def _stock_call(stock, att, q_x, kv_x, attn_bias, trunked_attn_bias, n_queries, n_keys, inf, inplace_safe, chunk_size):
    return stock(att, q_x, kv_x, attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias, n_queries=n_queries, n_keys=n_keys, inf=inf,
                 inplace_safe=inplace_safe, chunk_size=chunk_size)


# ----------------------------------------------------------------------------------------------------------------- dit_attn (+ dit_attn_fp16)
def _make_dit_forward(att, A, word: str, fp16_ok: bool):
    """primitives.Attention.forward for the 24 DiffusionTransformer token-attention modules: the provider's core by ``word``.  A call outside
    the site's contract (a local-attention call, no / per-sample / foreign pair bias) or a class no admitted row serves is answered by the
    module's own statement BY NAME, counted under ``aside`` -- never raised."""
    import torch
    H = att.num_heads
    st = STATE["dit_attn"]
    st16 = STATE["dit_attn_fp16"]
    stock = type(att).forward                                    # the module's own statement (class attribute: what --mode off runs at this site)

    def forward(q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
        args = (q_x, kv_x, attn_bias, trunked_attn_bias, n_queries, n_keys, inf, inplace_safe, chunk_size)
        st["calls"] += 1
        if n_queries or trunked_attn_bias is not None:                 # a local-attention call at the DiT token site: not this lever's op
            _count(st["aside"], "local_call"); return _stock_call(stock, att, *args)
        if attn_bias is None:
            _count(st["aside"], "no_bias"); return _stock_call(stock, att, *args)
        q = att.linear_q(q_x); k = att.linear_k(kv_x); v = att.linear_v(kv_x)
        g = att.linear_g(q_x) if att.gating else None
        C = q.shape[-1]; D = C // H
        lead = q.shape[:-2]; N = q.shape[-2]
        q4, k4, v4 = (t.reshape(-1, N, H, D) for t in (q, k, v))
        g4 = g.reshape(-1, N, H, D) if g is not None else None
        S = q4.shape[0]
        if k4.shape[0] != S:
            _count(st["aside"], "kv_samples"); return _stock_call(stock, att, *args)
        b = attn_bias
        while b.dim() > 3 and b.shape[0] == 1:
            b = b[0]
        if b.dim() != 3 or b.shape[0] != H or b.shape[-1] < N or b.shape[-2] < N:   # per-sample ([S,1,N,N] mask-only) or foreign bias: the statement's own path
            _count(st["aside"], "bias_shape"); return _stock_call(stock, att, *args)
        if b.shape[-1] != N or b.shape[-2] != N:                        # a padded row pitch (the sampler's 8-aligned bias planes): the N x N window
            b = b[:, :N, :N]
        if b.dtype != torch.float32:
            b = b.float()
        if b.stride(-1) != 1:
            b = b.contiguous()
        cap = _capturing(q4)
        try:
            sel = _select(A, "dit_attn", q4, N=N, S=S, heads=H, head_dim=D, word=word, capture=cap, fp16_ok=fp16_ok)
            o, sel = A.pair_bias_attention(q4, k4, v4, b[None], None, g4, word=word, selection=sel, scale=1.0 / math.sqrt(D), layout="snhd",
                                           cell=CELLS["dit_attn"], capture=cap, out_dtype=q.dtype)
        except A.Refusal as e:                                           # no admitted row serves this class: the statement itself, BY NAME (counted; never silent)
            _count(st["aside"], str(e.kind)); return _stock_call(stock, att, *args)
        _count(st["served"], A.arm_word(sel.row, sel.variant))
        if (sel.variant or "") == "fp16":
            st16["engaged_calls"] += 1
        return att.linear_o(o.reshape(*lead, N, C))

    forward.__wrapped__ = stock
    forward._fpf_apb = "dit_attn"
    return forward


def install_dit_attn(model, *, word: str, fp16: bool = False) -> Dict[str, Any]:
    """Bind the 24 DiffusionTransformer token-attention modules (primitives.Attention, 16 heads x 48) to the provider by ``word``; ``fp16`` =
    lever dit_attn_fp16 requested (rides: the fp16-operand arms are admissible).  Returns the lever's record; raises by name when the model
    does not have this topology (apb_levers answers an install failure with a stated aside: the modules keep the stock statement)."""
    from protenix.model.modules import primitives as P
    st = STATE["dit_attn"]
    st16 = STATE["dit_attn_fp16"]
    A = face()
    dm = _diffusion_module(model)
    mods = [blk.attention_pair_bias.attention for blk in dm.diffusion_transformer.blocks]
    if len(mods) != 24 or any(not isinstance(m, P.Attention) for m in mods):
        raise RuntimeError(f"dit_attn: expected 24 primitives.Attention token modules under diffusion_transformer.blocks, found {len(mods)}")
    for att in mods:
        if att.num_heads != DIT_HEADS or att.c_hidden != DIT_HEAD_DIM:   # primitives.Attention: c_hidden IS the per-head width (linear_q: c_hidden * num_heads)
            raise RuntimeError(f"dit_attn: token attention geometry heads={att.num_heads} c_hidden={att.c_hidden} (the provider cell {CELLS['dit_attn']} is 16 x 48)")
        if "forward" in vars(att) and not getattr(att.forward, "_fpf_apb", False) and not getattr(att, "_fpf_apb", False):
            raise RuntimeError("dit_attn: an Attention instance already carries an instance-level forward from another lever")
    for att in mods:
        att.forward = _make_dit_forward(att, A, word, fp16_ok=bool(fp16))
        att._fpf_apb = "dit_attn"                                # the ownership mark the sampler's fused-stack levers and the dtype gate read (name kept)
    st16["on"] = bool(fp16)
    st.update(installed_on=len(mods), cell_key=_device_cc_word(), precision=("per_cell" if fp16 else "fp32_class"),
              cell={"word": word, "face": "kernels.apb", "geometry": CELLS["dit_attn"], "fp16_arms": ("admissible" if fp16 else "off")})
    return dict(st, aside=None, left_alone=0)


# ----------------------------------------------------------------------------------------------------------------- atom_attn
def _make_atom_forward(att, A, word: str):
    """primitives.Attention.forward for the atom encoder / decoder local attention (local_cross_attention, 4 heads x 32, n_queries x n_keys
    windows, the windowed pair bias shared by the samples): the provider's windowed face by ``word``.  A call outside the contract or a class
    no admitted row serves is answered by the module's own statement BY NAME, counted under ``aside`` -- never raised."""
    import torch
    H = att.num_heads
    st = STATE["atom_attn"]
    stock = type(att).forward

    def forward(q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
        args = (q_x, kv_x, attn_bias, trunked_attn_bias, n_queries, n_keys, inf, inplace_safe, chunk_size)
        st["calls"] += 1
        if not (n_queries and n_keys) or trunked_attn_bias is None or attn_bias is not None:   # a global call at the atom site: not this lever's op
            _count(st["aside"], "global_call"); return _stock_call(stock, att, *args)
        q = att.linear_q(q_x); k = att.linear_k(kv_x); v = att.linear_v(kv_x)
        g = att.linear_g(q_x) if att.gating else None
        C = q.shape[-1]; D = C // H
        lead = q.shape[:-2]; N = q.shape[-2]
        q4, k4, v4 = (t.reshape(-1, N, H, D) for t in (q, k, v))
        g4 = g.reshape(-1, N, H, D) if g is not None else None
        if k4.shape[0] != q4.shape[0] or k4.shape[1] != N:
            _count(st["aside"], "kv_shape"); return _stock_call(stock, att, *args)
        b = trunked_attn_bias
        while b.dim() > 4:
            if b.shape[0] != 1:                                          # a per-sample local pair bias (the model's is sample-invariant [1, H, n_windows, n_queries, n_keys])
                _count(st["aside"], "per_sample_bias"); return _stock_call(stock, att, *args)
            b = b[0]
        if b.dim() != 4 or b.shape[0] != H or tuple(b.shape[-2:]) != (int(n_queries), int(n_keys)):
            _count(st["aside"], "bias_shape"); return _stock_call(stock, att, *args)
        if b.dtype != torch.float32:
            b = b.float()
        if b.stride(-1) != 1:
            b = b.contiguous()
        cap = _capturing(q4)
        try:
            sel = _select(A, "atom_attn", q4, N=N, S=q4.shape[0], heads=H, head_dim=D, word=word, capture=cap)
            o, sel = A.atom_attention(q4, k4, v4, b, g4, word=word, selection=sel, n_queries=int(n_queries), n_keys=int(n_keys),
                                      scale=1.0 / math.sqrt(D), capture=cap, out_dtype=q.dtype)
        except A.Refusal as e:
            _count(st["aside"], str(e.kind)); return _stock_call(stock, att, *args)
        _count(st["served"], A.arm_word(sel.row, sel.variant))
        return att.linear_o(o.reshape(*lead, N, C))

    forward.__wrapped__ = stock
    forward._fpf_apb = "atom_attn"
    return forward


def install_atom_attn(model, *, word: str) -> Dict[str, Any]:
    """Bind the 6 atom-transformer attention modules of the DiffusionModule's atom encoder + decoder (primitives.Attention with
    local_attention_method 'local_cross_attention', 4 heads x 32: the provider's windowed cell) by ``word``; raises by name on another topology."""
    from protenix.model.modules import primitives as P, transformer as T
    st = STATE["atom_attn"]
    A = face()
    dm = _diffusion_module(model)
    mods = []
    for part in (dm.atom_attention_encoder, dm.atom_attention_decoder):
        for m in part.modules():
            if isinstance(m, T.AtomTransformer):
                mods += [blk.attention_pair_bias.attention for blk in m.diffusion_transformer.blocks]
    if len(mods) != 6 or any(not isinstance(m, P.Attention) for m in mods):
        raise RuntimeError(f"atom_attn: expected 6 primitives.Attention modules under the diffusion module's atom encoder/decoder AtomTransformers, found {len(mods)}")
    for att in mods:
        meth = getattr(att, "local_attention_method", None)
        if meth != ATOM_METHOD:
            raise RuntimeError(f"atom_attn: local_attention_method={meth!r} (the provider's windowed cell serves {ATOM_METHOD!r})")
        if att.num_heads != ATOM_HEADS or att.c_hidden != ATOM_HEAD_DIM:
            raise RuntimeError(f"atom_attn: atom attention geometry heads={att.num_heads} c_hidden={att.c_hidden} (the provider cell {CELLS['atom_attn']} is 4 x 32)")
        if "forward" in vars(att) and not getattr(att.forward, "_fpf_apb", False) and not getattr(att, "_fpf_apb", False):
            raise RuntimeError("atom_attn: an Attention instance already carries an instance-level forward from another lever")
    for att in mods:
        att.forward = _make_atom_forward(att, A, word)
        att._fpf_apb = "atom_attn"
    st.update(installed_on=len(mods), cell_key=_device_cc_word(), cell={"word": word, "face": "kernels.apb", "geometry": CELLS["atom_attn"]})
    return dict(st, aside=None, left_alone=0)


# ----------------------------------------------------------------------------------------------------------------- pf_attn
def _big_bias(apb_mod, zz):
    """The memory line's chunked pair-bias producer for this module (big.apb_bias_chunked: the stock LayerNorm(z) + Linear on `rows` token
    rows at a time, engaged above the apb_bias_chunk lever's token gate) -> [H, N, N]; None outside big / below the gate."""
    B = sys.modules.get(__package__ + ".big") if __package__ else None
    fn = getattr(B, "apb_bias_chunked", None) if B is not None else None
    return fn(apb_mod, zz) if fn is not None else None


def _make_pf_smha(apb_mod, A, word: str):
    """AttentionPairBias.standard_multihead_attention for a Pairformer-style module (has_s=False self-attention): the provider's PRODUCER
    (LayerNorm(z) + Linear c_z->H, head-major planes) + CORE (gate fused) on the stock projections, by ``word``.  A call outside the contract
    (efficient fusion, cross-attention, a batched pair) or a class no admitted row serves takes the module's own statement BY NAME, counted."""
    import torch
    att = apb_mod.attention
    st = STATE["pf_attn"]
    stock = type(apb_mod).standard_multihead_attention
    cache: dict = {}                                             # the producer row's packed weights, per module (pair_bias_planes' cache contract)

    def smha(q, kv, z, inplace_safe=False, enable_efficient_fusion=False):
        st["calls"] += 1
        if enable_efficient_fusion or kv is not q:
            _count(st["aside"], "fusion" if enable_efficient_fusion else "cross_attention")
            return stock(apb_mod, q, kv, z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
        lead = q.shape[:-2]
        N = q.shape[-2]
        zz = z.reshape(-1, *z.shape[-3:])
        if zz.shape[0] != 1 or zz.shape[1] != N or zz.shape[2] != N:       # batched pair biases are not this site's contract
            _count(st["aside"], "pair_shape")
            return stock(apb_mod, q, kv, z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
        zz = zz[0]
        if zz.stride(2) != 1 or zz.stride(0) != N * zz.stride(1):
            zz = zz.contiguous()
        ln = apb_mod.layernorm_z
        H, D = att.num_heads, att.c_hidden
        a2 = q.reshape(-1, N, q.shape[-1])
        S = a2.shape[0]
        try:
            chunked = _big_bias(apb_mod, zz)                   # big above apb_bias_chunk's gate: the stock producer on `rows` token rows at a time (no full-N LayerNorm copy of the pair)
            if chunked is not None:
                b4 = chunked[None] if chunked.dim() == 3 else chunked
                if b4.dtype != torch.float32:
                    b4 = b4.float()
                prow = "apb_bias_chunk"
            else:
                psel = _select(A, "pf_planes", zz, N=N, S=1, heads=H, head_dim=D, word=word, capture=False, cell="bias_c%dh%d" % (int(zz.shape[-1]), H), c_z=int(zz.shape[-1]))
                planes, psel = A.pair_bias_planes(zz, ln.weight, getattr(ln, "bias", None), apb_mod.linear_nobias_z.weight, word=word, selection=psel,
                                                  eps=getattr(ln, "eps", 1e-5), out_layout="hij", out_dtype=torch.float32, cache=cache)
                b4 = planes if planes.dim() == 4 else planes[None]   # [1, H, N, ld] head-major planes (any row pitch >= N)
                prow = A.arm_word(psel.row, psel.variant)
            if b4.shape[-1] != N:
                b4 = b4[..., :N]
            qh = att.linear_q(a2).view(S, N, H, D)
            kh = att.linear_k(a2).view(S, N, H, D)
            vh = att.linear_v(a2).view(S, N, H, D)
            gh = att.linear_g(a2).view(S, N, H, D) if att.gating else None
            cap = _capturing(qh)
            csel = _select(A, "pf_attn", qh, N=N, S=S, heads=H, head_dim=D, word=word, capture=cap, timing=_timing(word, cap))   # the graph column for a call the stack graph plans to capture
            o, csel = A.pair_bias_attention(qh, kh, vh, b4, None, gh, word=word, selection=csel, scale=1.0 / math.sqrt(D), layout="snhd",
                                            cell=CELLS["pf_attn"], capture=cap, out_dtype=qh.dtype)
        except A.Refusal as e:
            _count(st["aside"], str(e.kind))
            return stock(apb_mod, q, kv, z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
        _count(st["producer"], prow)
        _count(st["served"], A.arm_word(csel.row, csel.variant))
        return att.linear_o(o.reshape(*lead, N, H * D))

    smha.__wrapped__ = stock
    smha._fpf_apb = "pf_attn"                                    # the ownership mark dtype_gate.py and the other levers read (name kept)
    return smha


def install_pf_attn(model, *, word: str) -> Dict[str, Any]:
    """Instance-level standard_multihead_attention on every Pairformer AttentionPairBias (has_s=False, 16 gated heads) OUTSIDE the
    DiffusionModule: the trunk PairformerStack (48 blocks) and the confidence head's pairformer (4 blocks).  has_s / cross-attention modules are
    left alone (counted); raises by name on another topology (apb_levers answers with a stated aside)."""
    from protenix.model.modules import transformer as T, diffusion as Dm
    st = STATE["pf_attn"]
    A = face()
    dm_ids = set()
    for m in model.modules():
        if isinstance(m, Dm.DiffusionModule):
            dm_ids |= {id(x) for x in m.modules()}
    mods, skipped_has_s = [], 0
    for name, m in model.named_modules():
        if not isinstance(m, T.AttentionPairBias) or id(m) in dm_ids:
            continue
        if m.has_s or getattr(m, "cross_attention_mode", False):
            skipped_has_s += 1
            continue
        att = m.attention
        if att.num_heads != PF_HEADS or att.c_hidden not in (16, 24, 32, 48, 64) or not att.gating:
            raise RuntimeError(f"pf_attn: {name}: heads={att.num_heads} c_hidden={att.c_hidden} gating={att.gating} (the provider cells are 16 gated heads)")
        c_z = m.linear_nobias_z.in_features
        if c_z & (c_z - 1) or c_z < 16:
            raise RuntimeError(f"pf_attn: {name}: c_z={c_z} (the producer rows need a power-of-two c_z >= 16)")
        if "standard_multihead_attention" in vars(m) and not getattr(m.standard_multihead_attention, "_fpf_apb", False):
            raise RuntimeError(f"pf_attn: {name}: the instance already carries a standard_multihead_attention from another lever")
        mods.append(m)
    if not mods:
        raise RuntimeError("pf_attn: no Pairformer AttentionPairBias (has_s=False) modules found outside the DiffusionModule")
    for m in mods:
        m.standard_multihead_attention = _make_pf_smha(m, A, word)
    st.update(installed_on=len(mods), skipped_has_s=skipped_has_s, cell_key=_device_cc_word(),
              cell={"word": word, "face": "kernels.apb", "geometry": CELLS["pf_attn"], "producer": "pair_bias_planes"})
    return dict(st, aside=None, left_alone=0)


def _device_cc_word() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return _cc_word(torch.cuda.get_device_capability())
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


INSTALLERS: Dict[str, Callable[..., Dict[str, Any]]] = {"dit_attn": install_dit_attn, "atom_attn": install_atom_attn, "pf_attn": install_pf_attn}


# ----------------------------------------------------------------------------------------------------------------- dit_attn_exact (hook contract)
def _exact_probe(A, stack: Optional[str], abi: Optional[str]) -> Dict[str, str]:
    """What the exact word resolves to on THIS stack for the DiT classes the sampler produces (S=5, eager + graph, the table's token buckets):
    {"<N>/<eager|graph>": arm}.  Pure table reads (no GPU work)."""
    out: Dict[str, str] = {}
    cc = _device_cc_tuple()
    for n in (256, 400, 800, 1200, 1536, 2048):
        for capture in (False, True):
            try:
                s = A.select(cc, "fp32", CELLS["dit_attn"], n, word="exact", samples=5, capture=capture, heads=DIT_HEADS, head_dim=DIT_HEAD_DIM,
                             stack=stack, abi=abi)
                out[f"{n}/{'graph' if capture else 'eager'}"] = A.arm_word(s.row, s.variant)
            except A.Refusal as e:
                out[f"{n}/{'graph' if capture else 'eager'}"] = "refused:" + str(e.kind)
    return out


def _device_cc_tuple() -> tuple:
    try:
        import torch
        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability())
    except Exception:  # noqa: BLE001
        pass
    return (0, 0)


def apply() -> Optional[str]:
    """Install the dit_attn_exact route when PTX_DIT_ATTN_EXACT=1 (else None, nothing touched).  Returns the ``DITATTN:on(...)`` marker; raises
    RuntimeError("dit_attn_exact: <reason>") when it cannot run here (the hook prints DITATTN:unavailable and the kit refuses the mode by name)."""
    if os.environ.get("PTX_DIT_ATTN_EXACT", "0") != "1":
        return None
    st = STATE["dit_attn_exact"]
    if st["installed"]:
        return st["marker"]
    import torch
    try:
        A = face()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{EXACT_LEVER}: {e}") from e
    if not torch.cuda.is_available():
        raise RuntimeError(f"{EXACT_LEVER}: CUDA is not available")
    stack = A.stack_word()
    abi = A.dit_exact_abi()
    probe = _exact_probe(A, stack, abi)
    kernel_classes = sorted(k for k, v in probe.items() if v == "dit_exact")
    mod = None
    loadcheck = None
    version = None
    if kernel_classes:                                           # the table vouches the kernel on this stack for some class: load + check it now (a failure refuses by name)
        try:
            DX = A.carried_module("dit_exact")
        except A.Refusal as e:
            raise RuntimeError(f"{EXACT_LEVER}: the provider's dit_exact row has no module here ({e})") from e
        try:
            mod = DX.load_prebuilt()                             # manifest (torch, CUDA, arch, .so + csrc sha256) + the 3-case in-process bit check vs torch's SDPA; raises by name
        except RuntimeError as e:
            raise RuntimeError(str(e) if str(e).startswith(EXACT_LEVER) else f"{EXACT_LEVER}: {e}") from e
        _DX["pkg"] = DX
        _DX["mod"] = mod
        loadcheck = dict(getattr(DX, "STATS", {}).get("loadcheck") or {})
        version = getattr(mod, "version", None) or (getattr(mod, "_dit_manifest", None) or {}).get("kernel_version")
    import protenix.model.modules.primitives as PR
    orig = PR._attention
    _DX["orig"] = orig
    cc = _device_cc_tuple()

    def _attention(q, k, v, attn_bias=None, use_efficient_implementation=True, inplace_safe=False):
        st["calls"] += 1
        if not use_efficient_implementation:
            why = "not_sdpa"
        elif attn_bias is None:
            why = "no_bias"
        elif q.dim() != 4:
            why = "rank"
        else:
            q32 = q if q.dtype == torch.float32 else q.to(dtype=torch.float32)   # the upcast the statement itself performs before SDPA
            k32 = k if k.dtype == torch.float32 else k.to(dtype=torch.float32)
            b32 = attn_bias if attn_bias.dtype == torch.float32 else attn_bias.to(dtype=torch.float32)
            B, H, N, D = q32.shape
            if mod is None:
                why = "exact_word:no_vouched_class"
            else:
                why = _DX["pkg"]._route(q32, k32, v, b32)      # the kernel's own envelope (head dim 48, strides, alignment, CUDA fp32): a declared route, decided before the call
                if why is None:
                    try:
                        sel = _select(A, "dit_attn_exact", q32, N=N, S=B, heads=H, head_dim=D, word="exact",
                                      capture=_capturing(q32), stack=stack, abi=abi, cell=CELLS["dit_attn"])
                        why = None if sel.row == "dit_exact" else "exact_word:" + A.arm_word(sel.row, sel.variant)
                    except A.Refusal as e:
                        why = "exact_word:refused:" + str(e.kind)
            if why is None:
                _count(st["routes"], "kernel")
                return mod.forward(q32, k32, v, b32, 1)          # scale 1: the statement's q arrives pre-scaled; bytes = the memory-efficient SDPA kernel's (vouched per stack)
        _count(st["routes"], why)
        return orig(q, k, v, attn_bias=attn_bias, use_efficient_implementation=use_efficient_implementation, inplace_safe=inplace_safe)

    PR._attention = _attention
    st.update(installed=True, loadcheck=loadcheck, row=("dit_exact" if kernel_classes else "statement"), stack=stack, abi=abi)
    dev = torch.cuda.get_device_name(0)
    lc = loadcheck or {}
    if kernel_classes:
        served = f"dit_exact v{version or '?'} {abi} serves {len(kernel_classes)}/{len(probe)} DiT classes (S5 N256..2048 eager|graph) vouched on {stack}"
        floor = sorted({v for v in probe.values() if v != "dit_exact"})
        served += (f"; the rest keep the statement by name ({','.join(floor)})" if floor else "")
        check = (f"loadcheck={lc.get('cases', 0)}cases{'-bit-equal' if lc.get('bit_equal', lc.get('ok', True)) else '-MISMATCH'} "
                 f"digests={'match' if lc.get('digest_match') else 'n/a'}")
    else:
        served = f"exact word -> {','.join(sorted(set(probe.values())))} on {stack} (no class vouched for dit_exact here): every call keeps the statement BY NAME"
        check = "loadcheck=none(kernel not loaded)"
    st["marker"] = (f"DITATTN:on({FACE} {core_version()} word=exact: {served}; {check}; {dev} cc={_cc_word(cc)}; "
                    f"EXACT-BITWISE vs mem-efficient SDPA fp32 D={DIT_HEAD_DIM})")
    return st["marker"]


def report() -> Dict[str, Any]:
    """dit_attn_exact's end-of-run record (the hook's _DXA_REPORT contract): unit, calls, routes, installed, loadcheck, row, stack."""
    st = STATE["dit_attn_exact"]
    return {"unit": EXACT_LEVER, "calls": int(st["calls"]), "routes": dict(st["routes"]), "installed": bool(st["installed"]),
            "loadcheck": st["loadcheck"], "row": st["row"], "stack": st["stack"], "abi": st["abi"], "face": FACE, "core": core_version()}


def lever_report() -> Dict[str, Dict[str, Any]]:
    """{lever: record} for the runner-seam levers (apb_levers.package_report's contract: installed_on, calls, cell, cell_key, named, error,
    precision + the served / aside / producer tallies as extra counters)."""
    out: Dict[str, Dict[str, Any]] = {}
    for lever in LEVERS:
        st = STATE[lever]
        rec = {k: st.get(k) for k in ("installed_on", "calls", "cell", "cell_key", "named", "error")}
        if lever == "dit_attn":
            rec["precision"] = st.get("precision")
            if st.get("left_alone"):                                 # modules another lever owns, left alone and NAMED (absent = none: the LEVER line carries no pair for it)
                rec["left_alone"] = list(st["left_alone"])
        rec["served"] = dict(st.get("served") or {})
        rec["aside_kinds"] = dict(st.get("aside") or {})
        if lever == "pf_attn":
            rec["producer"] = dict(st.get("producer") or {})
            rec["skipped_has_s"] = int(st.get("skipped_has_s") or 0)
        out[lever] = rec
    out["dit_attn_fp16"] = dict(STATE["dit_attn_fp16"])
    return out


def _reset_for_tests() -> None:
    global STATE
    STATE.clear()
    STATE.update(_fresh_state())
    _SEL.clear()
    _DX.update(mod=None, orig=None, pkg=None)
