"""Row schedules of the diffusion transformer's conditioned sub-layers on the core's row kernels (`opt_core.kernels.dtk_kernels`) — the
functions the DiT cells share (not a lever itself).

The token diffusion transformer's activations `a` are [.., S, N, c_a] rows (S samples) and its conditioning `s` is [.., 1, N, c_s]: ONE row
per token serving every sample. Stock broadcasts `s`-side results over the samples inside elementwise ops; these schedules hand the
conditioning rows to the kernels as PERIODIC operands (row r of the S·N activation rows reads conditioning row r % N — never expanded):

    adaln_forward(m, a, s)             AdaLN (layers: `AdaLN`: sigmoid(linear_g(LN_s s)) * LN_a(a) + linear_s(LN_s s)) as: LN_s (scale
                                       only) by `ln_modulate`, ONE GEMM s_n @ [W_g | W_s]^T (+ [b_g | 0]) for the gate logits and the shift,
                                       ONE `ln_modulate(a, scale, shift)` pass (sigmoid on the gate inside). Output in the compute dtype.
    cond_transition_forward(m, a, s, mask, residual=)   ConditionedTransitionBlock as: adaln rows -> ONE GEMM x @ [W_a | W_b]^T -> `swiglu`
                                       -> GEMM @ W_out^T -> `gate_residual(u, gate=s @ W_g^T + b_g (periodic), rowmask=mask (periodic),
                                       res=a if residual)`: the adaLN-zero output gate, the token mask and (optionally) the caller's residual
                                       add in one pass; output dtype = stock's promotion.

`plan_adaln` / `plan_cond_transition` state the served domain BY NAME (CUDA tensors, eval mode, fp32/bf16/fp16, the module layouts these
schedules read: LN_a without affine, LN_s scale-only, linear_g with bias, linear_s / swiglu / linear_out without, no Linear precision
override) and return (ok, reason, mode); a caller runs the stock module on a refusal. Weights are packed once per module × compute dtype
(re-packed when a parameter's storage or version changes). The compute dtype is the ambient CUDA autocast dtype when autocast is on (the
bf16 rollout), else `a`'s dtype; the arithmetic inside runs with autocast disabled on operands already in that dtype (fp32 accumulation in
the GEMMs and kernels). Numerics: tier 2 (bf16 rounding points differ from the stock op sequence; within the fast line's own bf16 error of
an fp64 evaluation at every measured step).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

KERNEL = "dtk_kernels"
STATS: Dict[str, Any] = {"packed": 0}
_DK = {"mod": None}


def _dtk():
    if _DK["mod"] is None:
        import importlib
        _DK["mod"] = importlib.import_module(KERNEL)          # routed from the core by the installing cell (opt_core.kernels.route)
    return _DK["mod"]


APB_CORE_MODULE = "opt_core.attn.apb_core"                        # the tree's pair-bias attention entry: pair_bias_attention(q, k, v, pair_bias, key_mask, *, gate, num_samples, num_heads, layout, out, scale, inf), Unsupported(.event), census(), kernel()
APB_PROVIDER_MODULE = "opt_core.kernels.apb"                     # the core's ONE pair-bias attention provider: select(cc, dtype, cell, n, word=) + pair_bias_attention(..., word=) over every carried row and the measured cell table
_APB_WORD: Dict[str, Any] = {"word": None, "source": None}         # the process's bound provider word (set_apb_word) — None: the direct entry serves (resolve_apb_core)


def set_apb_word(word: Optional[str], source: Optional[str] = None) -> Optional[str]:
    """Bind the pair-bias attention core of every cell that resolves it with pref "auto" (the DiT attention levers, the fused DiT block, the
    trunk's attention with pair bias) to the core's provider asked by `word`: a tier word (fast | big | exact | faithful: the measured cell
    table's row per (capability, dtype, geometry, samples, token bucket, eager | graph timing)) or any provider row word (`row[:variant]`, an
    ablation by name). None unbinds (the direct entry again). Raises ValueError on a word the provider does not know. Cells resolve their core
    once, at their first served call: bind before the first forward. Returns the bound word."""
    if word is None or not str(word).strip():
        _APB_WORD.update(word=None, source=source)
        return None
    w = provider_word(word)
    _APB_WORD.update(word=w, source=source)
    return w


def provider_word(word) -> str:
    """`word` normalised (lower case, stripped) when the provider knows it — a tier word, or `row[:variant]` with a row of the provider's and a
    variant that row lists — else ValueError naming the vocabulary."""
    import importlib
    P = importlib.import_module(APB_PROVIDER_MODULE)
    w = str(word).strip().lower()
    rw, variant = P.split_word(w)
    if rw in P.TIER_WORDS and variant is None:
        return w
    if rw not in P.ROW_NAMES:
        raise ValueError("%r is not a pair-bias attention provider word (tier words %s; rows %s)" % (word, "|".join(P.TIER_WORDS), "|".join(P.ROW_NAMES)))
    if variant is not None and variant not in tuple(getattr(P, "VARIANTS", {}).get(rw, ())):
        raise ValueError("%r: row %s has no variant %r (variants: %s)" % (word, rw, variant, "|".join(getattr(P, "VARIANTS", {}).get(rw, ())) or "none"))
    return w


def apb_word() -> Optional[str]:
    """The bound provider word or None."""
    return _APB_WORD["word"]


class ProviderUnsupported(Exception):
    """A call the provider refused by name for the asked word with no row it names to take it (`.event`: the refusal kind's head, `.detail`)."""
    def __init__(self, event: str, detail: str = ""):
        super().__init__("%s: %s" % (event, detail) if detail else event)
        self.event, self.detail = event, detail


PER_SAMPLE_ROW = "dtk_loop"          # kernels.apb's row that takes one bias plane set / key mask PER SAMPLE (the per-sample flash-bias loop); the fused rows take one shared set


class ProviderCore:
    """The direct entry's calling convention (`pair_bias_attention(q, k, v, pair_bias, key_mask, *, gate, num_samples, num_heads, layout, out,
    scale, inf)`, `Unsupported`, `census()`, `kernel()`) over the core's pair-bias attention PROVIDER asked by ONE word.

    Per call class (dtype, samples, bias batch, heads, head dim, tokens, mask presence, eager | capturing) the provider's `select` runs once
    (memoised; the provider records the cell census there) and its choice serves every later call of the class through `selection=`. Timing
    model per call: a call made while the current stream is capturing asks the graph-replay cells (`capture=True`), any other call the eager
    cells; the first eager call of a class also launches the capture-time row once on the same operands when it differs (its JIT compile and
    module load then happen outside any capture — the graphed lines warm up eagerly before they capture). A refusal that names a row is taken
    ONCE by that name for the call class (counted `fallback`), a refusal naming none raises `Unsupported` (the caller's stock path, counted
    there). The bias is one plane set for all samples ([H, N, N] / [1, H, N, N]) or one per sample ([S, H, N, N]; rows that take only the
    shared form refuse by name and name the row that takes it). `inf` is accepted for the convention; the provider
    folds the key mask with its own additive constant (-1e9 on dropped keys), `scale` None = head_dim ** -0.5 (as the entry)."""
    Unsupported = ProviderUnsupported
    MEMO_CAP = 64

    def __init__(self, word: str):
        import importlib
        self.P = importlib.import_module(APB_PROVIDER_MODULE)
        self.word = provider_word(word)
        w = self.word
        self._sel: Dict[tuple, Any] = {}
        self._warmed: set = set()
        self._stack = None
        self.CENSUS: Dict[str, Any] = {"word": w, "served": 0, "rows": {}, "cells": {}, "timing": {"eager": 0, "graph": 0}, "fallback": {}, "refused": {}, "prewarm": {}}

    # -- the entry's small surface
    def kernel(self):
        return self.P

    def census(self) -> dict:
        c = self.CENSUS
        return {"word": c["word"], "served": c["served"], "rows": dict(c["rows"]), "cells": dict(c["cells"]), "timing": dict(c["timing"]),
                "fallback": dict(c["fallback"]), "refused": dict(c["refused"]), "prewarm": dict(c["prewarm"])}

    @staticmethod
    def _bump(d: dict, key, cap: int = 24) -> None:
        if key in d or len(d) < cap:
            d[key] = d.get(key, 0) + 1
        else:
            d["…"] = d.get("…", 0) + 1

    def _cell(self, H: int, D: int):
        for kind in ("dit", "pf", "msarow"):
            w = self.P.cell_word(kind, heads=H, head_dim=D)
            if w is not None:
                return w
        return None

    def _stack_word(self, device):
        if self._stack is None:
            try:
                self._stack = self.P.stack_word(device)
            except Exception:  # noqa: BLE001
                self._stack = ""
        return self._stack or None

    def _select(self, q, S: int, H: int, D: int, N: int, capture: bool, word: Optional[str] = None):
        import torch
        w = word or self.word
        key = (w, q.dtype, S, H, D, N, bool(capture))
        sel = self._sel.get(key)
        if sel is None:
            cc = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
            rw = self.P.split_word(w)[0]
            abi = self.P.dit_exact_abi() if rw in ("dit_exact", "exact", "faithful") else None
            sel = self.P.select(cc, q.dtype, self._cell(H, D), N, word=w, samples=S, capture=bool(capture), head_dim=D, heads=H,
                                stack=self._stack_word(q.device) if q.is_cuda else None, abi=abi)
            if len(self._sel) >= self.MEMO_CAP:
                self._sel.clear()
            self._sel[key] = sel
        return sel

    def _serve(self, q4, k4, v4, pb, km, g4, S_sel, H, D, N, lay, capture, scale, config, dtype, word: Optional[str] = None):
        """Select the row for this call class at the call's REAL sample count `S_sel` (the tensors may be one sample of it), serve, and on a
        by-name refusal serve the named fallback row; books served / rows / cells / timing / fallback / refused."""
        P = self.P
        w = word or self.word
        o = served = None
        try:
            sel = self._select(q4, S_sel, H, D, N, capture, word=(w if word else None))
            o, served = P.pair_bias_attention(q4, k4, v4, pb, km, g4, word=w, selection=sel, scale=scale, layout=lay, capture=capture, config=config)
        except P.Refusal as r:
            fb = r.fallback
            kind = str(r.kind).split("(")[0]
            self._bump(self.CENSUS["fallback"], "%s:%s->%s" % (r.row or w, kind, fb or "caller"))
            if not fb or P.split_word(fb)[0] not in P.ROW_NAMES:
                self._bump(self.CENSUS["refused"], kind)
                raise ProviderUnsupported("provider:%s" % kind.split(":")[0], str(r))
            try:
                fsel = self._select(q4, S_sel, H, D, N, capture, word=fb)
                o, served = P.pair_bias_attention(q4, k4, v4, pb, km, g4, word=fb, selection=fsel, scale=scale, layout=lay, capture=capture, config=config)
                self._sel[(self.word, dtype, S_sel, H, D, N, capture)] = fsel      # the class keeps the named row from here (one refusal per class, not per call)
            except P.Refusal as r2:
                kind2 = str(r2.kind).split("(")[0]
                self._bump(self.CENSUS["refused"], kind2)
                raise ProviderUnsupported("provider:%s" % kind2.split(":")[0], str(r2))
        self.CENSUS["served"] += 1
        self._bump(self.CENSUS["rows"], P.arm_word(served.row, served.variant))
        self._bump(self.CENSUS["cells"], "%s%s" % (served.cell or "none", "" if served.size_measured else "~"))
        self.CENSUS["timing"]["graph" if capture else "eager"] += 1
        return o, served

    def pair_bias_attention(self, q, k, v, pair_bias, key_mask=None, *, gate=None, num_samples: Optional[int] = None, num_heads: Optional[int] = None,
                            layout: str = "rows", out=None, scale: Optional[float] = None, inf: float = 1e9, config: Optional[dict] = None):
        import torch
        P = self.P
        if layout == "shnd":                                                   # [S, H, N, D] views (strided is fine); gate [S, N, H*D] | [S, N, H, D]
            S, H, N, D = (int(x) for x in q.shape)
            q4, k4, v4, lay = q, k, v, "shnd"
            g4 = None if gate is None else gate.reshape(S, N, H, D).permute(0, 2, 1, 3)
        elif layout == "rows":                                                 # [S*N, H*D] column slices of one GEMM result (row stride >= H*D, unit column stride)
            if num_samples is None or num_heads is None:
                raise ProviderUnsupported("args", "layout rows needs num_samples and num_heads")
            S, H = int(num_samples), int(num_heads)
            R, HD = (int(x) for x in q.shape)
            N, D = R // S, HD // H
            try:
                q4, k4, v4 = q.view(S, N, H, D), k.view(S, N, H, D), v.view(S, N, H, D)
                g4 = None if gate is None else gate.view(S, N, H, D)
            except RuntimeError as e:
                raise ProviderUnsupported("layout", "rows operands not viewable as [S, N, H, D] (%s)" % str(e).split("\n")[0][:60])
            lay = "snhd"
        else:
            raise ProviderUnsupported("layout", "layout %r (rows | shnd)" % (layout,))
        pb = pair_bias
        if pb is not None:
            if pb.dim() < 3 or tuple(int(x) for x in pb.shape[-3:]) != (H, N, N):
                raise ProviderUnsupported("bias_shape", "pair bias %s for heads=%d tokens=%d" % (tuple(pb.shape), H, N))
            lead = 1
            for d_ in pb.shape[:-3]:
                lead *= int(d_)
            if lead not in (1, S):
                raise ProviderUnsupported("bias_shape", "pair bias batch %d for samples=%d (one shared plane set or one per sample)" % (lead, S))
            pb = pb.reshape(lead, H, N, N)
        km = key_mask
        if km is not None:
            if int(km.shape[-1]) != N or km.numel() % N or (km.numel() // N) not in (1, S):
                raise ProviderUnsupported("mask_shape", "key mask %s for samples=%d tokens=%d" % (tuple(km.shape), S, N))
            km = km.reshape(-1, N)
        if not q.is_cuda:
            raise ProviderUnsupported("device", "the provider's rows serve CUDA tensors (%s)" % (q.device,))
        capture = bool(torch.cuda.is_current_stream_capturing())
        if not capture:                                                        # launch the capture-time row of this class once, eagerly, when it differs (JIT / module load outside any capture)
            try:
                sel0, gsel = self._select(q4, S, H, D, N, False), self._select(q4, S, H, D, N, True)
                gkey = (P.arm_word(gsel.row, gsel.variant), q.dtype, H, D)
                if (gsel.row, gsel.variant) != (sel0.row, sel0.variant) and gkey not in self._warmed:
                    self._warmed.add(gkey)
                    try:
                        P.pair_bias_attention(q4, k4, v4, pb, km, g4, word=self.word, selection=gsel, scale=scale, layout=lay, capture=True, config=config)
                        self._bump(self.CENSUS["prewarm"], gkey[0])
                    except Exception as e:  # noqa: BLE001 — a row that cannot serve here is met again (and booked) at capture time
                        from opt_core.oom import is_oom
                        if is_oom(e): raise                                              # device out-of-memory during a pre-warm is the caller's to see, never a census token
                        self._bump(self.CENSUS["prewarm"], "%s:failed(%s)" % (gkey[0], type(e).__name__))
            except P.Refusal:
                pass                                                           # the asked word refuses this class by name: booked below on the serving path
        o = served = None
        if S > 1 and ((pb is not None and int(pb.shape[0]) == S) or (km is not None and int(km.shape[0]) == S)) \
                and self.P.split_word(self.word)[0] not in ("dit_exact", "exact", "faithful"):
            # a bias plane set or a key mask PER SAMPLE (the diffusion transformer's own pair bias / padding mask per sample; the provider folds a
            # per-sample mask into a per-sample plane set): the fused rows take ONE shared, key-contiguous plane set and assert it, so this call
            # class is served by the provider's per-sample row `PER_SAMPLE_ROW` (the loop over samples on the flash-bias kernel these callers ran
            # before the provider) in one call; exact-class words keep their own row.
            o, served = self._serve(q4, k4, v4, pb, km, g4, S, H, D, N, lay, capture, scale, config, q.dtype, word=PER_SAMPLE_ROW)
            self._bump(self.CENSUS["rows"], "per_sample:S%d" % S)
        else:
            o, served = self._serve(q4, k4, v4, pb, km, g4, S, H, D, N, lay, capture, scale, config, q.dtype)
        o4 = o if lay == "snhd" else o.permute(0, 2, 1, 3)                    # [S, N, H, D]
        if layout == "shnd":
            res = o4.reshape(S, N, H * D)
        else:
            res = o4.reshape(S * N, H * D)
        if out is not None:
            if tuple(out.shape) == tuple(res.shape):
                out.copy_(res)
            else:
                out.copy_(res.reshape(out.shape))
            return out
        return res




def resolve_apb_core(pref: str = "auto"):
    """(core, "") for the pair-bias attention core the DiT / trunk cells call, or (None, reason).

    `pref`: "auto" = the core's ONE pair-bias attention PROVIDER (`opt_core.kernels.apb`) asked by the process's bound word when one is bound
    (`set_apb_word`: a tier word fast | big | exact | faithful, or any provider row word) — a `ProviderCore` with the direct entry's calling
    convention — else the direct entry `opt_core.attn.apb_core` with its carried kernel routed (nothing bound: every byte as before);
    "dtk" = (None, "pinned:dtk"): the caller serves with dtk_kernels.flash_bias_attn per sample; any other value = a provider word for THIS
    caller alone (e.g. "dtk_loop", "sdpa", "fast"). (None, reason) also when the entry / provider is not importable."""
    if pref == "dtk":
        return None, "pinned:dtk"
    word = _APB_WORD["word"] if pref in ("auto", None, "") else str(pref).strip().lower()
    if word:
        try:
            return ProviderCore(word), ""
        except Exception as e:  # noqa: BLE001
            return None, "provider:%s" % (str(e).split("\n")[0][:80].replace(" ", "_") or type(e).__name__)
    try:
        import importlib
        C = importlib.import_module(APB_CORE_MODULE)
        if not (hasattr(C, "pair_bias_attention") and hasattr(C, "Unsupported")):
            raise ImportError(f"{APB_CORE_MODULE} has no pair_bias_attention / Unsupported")
        if hasattr(C, "kernel"):
            C.kernel()                                             # routes the carried kernel now: an import failure is a reason here, not mid-prediction
        return C, ""
    except Exception as e:  # noqa: BLE001
        return None, "import:%s" % (getattr(e, "event", None) or type(e).__name__)


# ----------------------------------------------------------------------------------------------------------------- dtype / shape planning
def compute_dtype(a):
    """The dtype the GEMM operands take: the ambient CUDA autocast dtype when autocast is on, else a's own dtype."""
    import torch
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return a.dtype


def lead_period(lead_small, lead_big) -> Optional[int]:
    """Rows-period P if a tensor with leading shape `lead_small` broadcasts against `lead_big` over LEADING dims only (its rows then serve the
    prod(lead_big) rows periodically: row r <- row r % P); P == prod(lead_big) when the shapes are equal. None when the broadcast interleaves
    (a non-1 dim of small before a broadcast dim): the caller expands."""
    small = tuple(int(d) for d in lead_small); big = tuple(int(d) for d in lead_big)
    if len(small) > len(big):
        return None
    small = (1,) * (len(big) - len(small)) + small
    for j in range(len(big) + 1):
        if all(d == 1 for d in small[:j]) and small[j:] == big[j:]:
            return max(1, math.prod(big[j:]))
    return None


def _ok_dtypes():
    import torch
    return (torch.bfloat16, torch.float16, torch.float32)


def _linear_ok(lin, want_bias: bool) -> Optional[str]:
    if getattr(lin, "precision", None) is not None:
        return "linear-precision"
    if (getattr(lin, "bias", None) is not None) != want_bias:
        return "bias-layout"
    return None


def plan_adaln(m, a, s) -> Tuple[bool, str, Optional[str]]:
    """(ok, reason, mode) for AdaLN module m on (a, s); mode = direct | periodic | expanded (how s's rows serve a's)."""
    import torch
    if not (torch.is_tensor(a) and a.is_cuda and torch.is_tensor(s) and s.is_cuda):
        return False, "not_cuda", None
    if m.training:
        return False, "training", None
    okd = _ok_dtypes()
    if a.dtype not in okd or s.dtype not in okd:
        return False, f"dtype:{a.dtype}/{s.dtype}".replace("torch.", ""), None
    if not all(hasattr(m, n) for n in ("layer_norm_a", "layer_norm_s", "linear_g", "linear_s", "c_a", "c_s")):
        return False, "not-adaln", None
    if a.shape[-1] != m.c_a or s.shape[-1] != m.c_s:
        return False, "channels", None
    ln_a, ln_s = m.layer_norm_a, m.layer_norm_s
    if getattr(ln_a, "weight", None) is not None or getattr(ln_a, "bias", None) is not None or getattr(ln_s, "bias", None) is not None or getattr(ln_s, "weight", None) is None:
        return False, "ln-layout", None
    for lin, wb in ((m.linear_g, True), (m.linear_s, False)):
        r = _linear_ok(lin, wb)
        if r:
            return False, r, None
    if a.numel() == 0:
        return False, "empty", None
    P = lead_period(s.shape[:-1], a.shape[:-1])
    return True, "", ("direct" if P == math.prod(a.shape[:-1]) else "periodic") if P is not None else "expanded"


def plan_cond_transition(m, a, s, mask) -> Tuple[bool, str, Optional[str]]:
    import torch
    ok, why, mode = plan_adaln(m.layer_norm, a, s) if hasattr(getattr(m, "layer_norm", None), "layer_norm_a") else (False, "not-adaln", None)
    if not ok:
        return ok, why, mode
    sw = getattr(m, "swiglu", None)
    if sw is None or not (hasattr(sw, "linear_a") and hasattr(sw, "linear_b")) or not hasattr(m, "linear_out") or not hasattr(m, "linear_g"):
        return False, "swiglu-layout", None
    for lin, wb in ((sw.linear_a, False), (sw.linear_b, False), (m.linear_out, False), (m.linear_g, True)):
        r = _linear_ok(lin, wb)
        if r:
            return False, r, None
    if mask is not None:
        if not (torch.is_tensor(mask) and mask.is_cuda):
            return False, "mask_not_cuda", None
        if int(mask.shape[-1]) != int(a.shape[-2]):
            return False, "mask_shape", None
    return True, "", mode


# ----------------------------------------------------------------------------------------------------------------- weights (packed once per module x dtype)
def _key(params, dtype, device):
    return tuple((p.data_ptr(), p._version) for p in params) + (str(dtype), str(device))


def adaln_weights(m, dtype, device) -> dict:
    import torch
    params = (m.linear_g.weight, m.linear_g.bias, m.linear_s.weight, m.layer_norm_s.weight)
    key = _key(params, dtype, device)
    cache = m.__dict__.get("_of3opt_adaln_w")
    if cache is None or cache["key"] != key:
        with torch.no_grad():
            w_gs = torch.cat([m.linear_g.weight.detach(), m.linear_s.weight.detach()], 0).to(device=device, dtype=dtype).contiguous()      # [2 c_a, c_s]
            b_gs = torch.cat([m.linear_g.bias.detach(), torch.zeros_like(m.linear_g.bias)], 0).to(device=device, dtype=dtype).contiguous()   # [2 c_a]
            w_lns = m.layer_norm_s.weight.detach().to(device=device, dtype=torch.float32).contiguous()
        cache = {"key": key, "w_gs": w_gs, "b_gs": b_gs, "w_lns": w_lns, "eps_a": float(m.layer_norm_a.eps), "eps_s": float(m.layer_norm_s.eps)}
        m.__dict__["_of3opt_adaln_w"] = cache; STATS["packed"] += 1
    return cache


def cond_weights(m, dtype, device) -> dict:
    import torch
    params = (m.swiglu.linear_a.weight, m.swiglu.linear_b.weight, m.linear_out.weight, m.linear_g.weight, m.linear_g.bias)
    key = _key(params, dtype, device)
    cache = m.__dict__.get("_of3opt_cond_w")
    if cache is None or cache["key"] != key:
        with torch.no_grad():
            w_ab = torch.cat([m.swiglu.linear_a.weight.detach(), m.swiglu.linear_b.weight.detach()], 0).to(device=device, dtype=dtype).contiguous()   # [2 h, c_a]
            w_o = m.linear_out.weight.detach().to(device=device, dtype=dtype).contiguous()                                                              # [c_a, h]
            w_gate = m.linear_g.weight.detach().to(device=device, dtype=dtype).contiguous()                                                              # [c_a, c_s]
            b_gate = m.linear_g.bias.detach().to(device=device, dtype=dtype).contiguous()
        cache = {"key": key, "w_ab": w_ab, "w_o": w_o, "w_gate": w_gate, "b_gate": b_gate}
        m.__dict__["_of3opt_cond_w"] = cache; STATS["packed"] += 1
    return cache


# ----------------------------------------------------------------------------------------------------------------- the ops
class Refuse(Exception):
    """A by-name refusal raised before any fused work; the caller runs the stock module and counts str(exc)."""


def _rows(t, c):
    t2 = t.reshape(-1, c)
    if t2.numel() and t2.stride(-1) != 1:
        t2 = t2.contiguous()
    return t2


def _s_rows(s, a, c_s, mode):
    """s as [P, c_s] rows + the period P serving a's rows (expanded to a's rows when the broadcast interleaves)."""
    if mode == "expanded":
        s = s.expand(*a.shape[:-1], c_s)
    s2 = _rows(s, c_s)
    return s2, s2.shape[0]


def adaln_rows(m, a2, s2, P: int, cd):
    """AdaLN on rows: a2 [R, c_a] (any float dtype), s2 [P, c_s]. Returns x [R, c_a] in cd."""
    import torch
    dtk = _dtk()
    W = adaln_weights(m, cd, a2.device)
    R = a2.shape[0]
    s_n = dtk.ln_modulate(s2, weight=W["w_lns"], eps=W["eps_s"], out_dtype=cd)                                  # LN_s (scale only) -> cd
    gs = torch.addmm(W["b_gs"], s_n, W["w_gs"].t())                                                                # [P, 2 c_a]: gate logits | shift
    c_a = m.c_a
    return dtk.ln_modulate(a2, scale=gs[:, :c_a], shift=gs[:, c_a:], eps=W["eps_a"], out_dtype=cd, sigmoid_scale=True,
                           mod_period=(P if P != R else None))                                                     # sigmoid(g) * LN(a) + shift, one kernel


def adaln_forward(m, a, s):
    """AdaLN(a, s) -> [.., N, c_a] in the compute dtype. Raises Refuse(reason) by name before any work when (m, a, s) is outside the domain."""
    import torch
    ok, why, mode = plan_adaln(m, a, s)
    if not ok:
        raise Refuse(why)
    cd = compute_dtype(a)
    with torch.autocast("cuda", enabled=False):
        a2 = _rows(a, m.c_a)
        s2, P = _s_rows(s, a, m.c_s, mode)
        x = adaln_rows(m, a2, s2, P, cd)
    return x.reshape(*a.shape[:-1], m.c_a)


def cond_transition_forward(m, a, s, mask=None, *, residual: bool = False):
    """ConditionedTransitionBlock(a, s, mask) [+ a when residual] -> [.., N, c_a] in stock's output dtype (promote(cd, mask or a) [, a]).
    Raises Refuse(reason) by name before any work when outside the domain."""
    import torch
    ok, why, mode = plan_cond_transition(m, a, s, mask)
    if not ok:
        raise Refuse(why)
    dtk = _dtk()
    cd = compute_dtype(a)
    c_a, c_s = m.c_a, m.c_s
    with torch.autocast("cuda", enabled=False):
        a2 = _rows(a, c_a)
        R = a2.shape[0]
        s2, P = _s_rows(s, a, c_s, mode)
        x = adaln_rows(m.layer_norm, a2, s2, P, cd)                                                                # [R, c_a] cd
        W = cond_weights(m, cd, a2.device)
        ab = x @ W["w_ab"].t()                                                                                     # [R, 2h] cd (fp32 accumulate in cuBLAS)
        b = dtk.swiglu(ab)                                                                                         # silu(a) * b -> [R, h] cd
        u = b @ W["w_o"].t()                                                                                       # [R, c_a] cd
        s2c = s2 if s2.dtype == cd else s2.to(cd)
        gg = torch.addmm(W["b_gate"], s2c, W["w_gate"].t())                                                       # [P, c_a] gate logits (pre-sigmoid)
        mk, PM = None, None
        out_dtype = torch.promote_types(cd, mask.dtype) if mask is not None else torch.promote_types(cd, a.dtype)  # stock: (sigmoid(g) * u) * mask (mask = ones in a's dtype when None)
        if mask is not None:
            PMl = lead_period(mask.shape, a.shape[:-1])
            if PMl is None:
                mk = mask.expand(*a.shape[:-1]).reshape(-1); PM = R
            else:
                mk = mask.reshape(-1); PM = PMl
            if mk.dtype == torch.bool:
                mk = mk.to(torch.float32)
        out = dtk.gate_residual(u, gate=gg, gate_period=P, rowmask=mk, mask_period=PM, res=(a2 if residual else None), sigmoid_gate=True,
                                out_dtype=(torch.promote_types(out_dtype, a.dtype) if residual else out_dtype))
    return out.reshape(*a.shape[:-1], c_a)
