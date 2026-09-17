"""fpf_rf3_adapter — install FlashPairformer (FPF) TriMul kernels into RoseTTAFold3 (foundry `rf3`, pin 4010e3e) WITHOUT editing files or weights.

RF3's TriangleMultiplication (rf3/model/layers/attention.py) in its default (cuEquivariance) path calls
    cuet.triangle_multiplicative_update(x=pair.bf16, direction, mask=None, norm_in.{w,b}, p_in.W[2D,C], g_in.W[2D,C], norm_out.{w,b}, p_out.W[C,D], g_out.W[C,C], eps=1e-5)
and the CALLER (PairformerBlock / MSAModule / template pairformer / confidence-head pairformer) adds the residual (dropout is a no-op at inference).
This is the AF3 layout with a|b concatenated projections: rows [0:D] of p_in/g_in = "left"/a, rows [D:2D] = "right"/b  (same convention as cuEq and as
RF3's own non-cuEq path).  Mapping onto FPF generic.trimul (fpf_trimul_v4):
    ln_in_w/b = norm_in.weight/bias [C];  w_ap = p_in.W[:D], w_bp = p_in.W[D:], w_ag = g_in.W[:D], w_bg = g_in.W[D:]  (no biases)
    ln_out_w/b = norm_out.weight/bias [D]; w_o = p_out.W [C,D]; w_og = g_out.W [C,C]; mask=None; residual=False (caller adds it); direction as the module's.
Served: d_pair == 128 and d_hidden == 128 (48 pairformer blocks x2, 4 MSA-module blocks x2, 4 confidence-head pairformer blocks x2), CUDA, bf16 autocast,
N from the token floor up: under the modes' TIER words (fast.fast / fast.big -> fpf_rf3_trimul_rows.serve_fast) the provider table's per-class floor
(fpf_rf3_trimul_rows.fast_floor = opt_core.kernels.trimul.v4_n_min: 2 tokens on the cc-9.0 c_z-128 classes whose small-N cells name row v4, 101 on every
other class); the default ROW word v4 (G.trimul direct, below) keeps the package's own N_MIN = 101 (no size ceiling — large N may exceed GPU memory: see
fpf_trimul_v4's notes for the workspace formula).
NOT served (stays on the model's own path, counted): the template-track pairformer (c = 64, 2 blocks x2 per recycle), N below the floor, any TrimulUnsupported
reason, and the cuEq-disabled configuration.
Modes:  'fast'  -> fpf_trimul_v4.generic (same numerics class as the cuEq fused TriMul: bf16 tensor-core GEMMs / fp32 accumulate / fp32 LN stats+gating; NOT bitwise to stock)
        'stock' -> original forward (disable()).
"""
import os, time, torch
import numpy as np
from opt_core.oom import is_oom          # an out-of-memory is the caller's to see: every reroute below re-raises it first (never a kernel error, never a fallback)

MODE = "stock"
COUNTS = {"served": 0, "fallback": {}, "errors": 0, "first": None}
_ORIG = {}
TRIMUL_PAD = 16          # fpf_trimul_v4 generic.trimul pad: token rows padded to a multiple of it


def _count_fb(reason):
    COUNTS["fallback"][reason] = COUNTS["fallback"].get(reason, 0) + 1


def _weights_fast(m):
    from fpf_trimul_v4 import generic as G
    D = m.d_hidden
    return G.pack_weights(ln_in_w=m.norm_in.weight.detach(), ln_in_b=m.norm_in.bias.detach(),
                          w_ag=m.g_in.weight.detach()[:D], w_ap=m.p_in.weight.detach()[:D],
                          w_bg=m.g_in.weight.detach()[D:], w_bp=m.p_in.weight.detach()[D:],
                          ln_out_w=m.norm_out.weight.detach(), ln_out_b=m.norm_out.bias.detach(),
                          w_o=m.p_out.weight.detach(), w_og=m.g_out.weight.detach(), cache_owner=m, cache_key="rf3_fast")


def _fast_forward(self, pair):
    from foundry import SHOULD_USE_CUEQUIVARIANCE
    from fpf_trimul_v4 import generic as G
    import fpf_rf3_trimul_rows as TR                      # the shared core's ONE trimul provider: the head sub-word 'fast.<word>' names the row (default word v4 = G.trimul below,
    if not (self.use_cuequivariance and SHOULD_USE_CUEQUIVARIANCE):   # byte for byte); any other word serves through the provider's face or declines BY NAME (TR.serve_fast)
        _count_fb("vanilla-path-not-served"); return _ORIG["forward"](self, pair)
    word = TR.STATE["fast"]["word"]
    today = word == TR.DEFAULT_WORD["fast"]
    if today and (self.d_pair != 128 or self.d_hidden != 128):
        _count_fb("c=%d/d=%d" % (self.d_pair, self.d_hidden)); return _ORIG["forward"](self, pair)
    x = pair
    if torch.is_autocast_enabled():
        x = x.to(dtype=torch.get_autocast_dtype(x.device.type))   # exactly what the stock cuEq branch does
    if today:
        w = _weights_fast(self)
        ok, why = G.supported(x, None, weights=w)
        if not ok:
            _count_fb(str(why)); return _ORIG["forward"](self, pair)
    try:
        _res = bool(getattr(self, "_fpf_res", False)) and x.dtype == torch.bfloat16
        if today:
            out = G.trimul(x, None, direction=self.direction, weights=w, residual=_res, pad=TRIMUL_PAD)
            TR.note_default(self, x)                         # the provider's SELECT line + row census for the default word (a table read; launches nothing, never raises)
        else:
            out = TR.serve_fast(self, x, residual=_res, pad=TRIMUL_PAD)
        self._fpf_res_done = _res
    except TR.Declined as d:    # no admitted row serves this call under the word (its floor, its widths, this card): the stock statement BY NAME, counted under the word
        _count_fb(d.word); return _ORIG["forward"](self, pair)
    except Exception as e:  # never crash a prediction — except on out-of-memory, which propagates
        if is_oom(e): raise
        COUNTS["errors"] += 1
        if COUNTS["errors"] == 1:
            print("[fpf_rf3] KERNEL ERROR (fast) -> stock for this call:", repr(e)[:300], flush=True)
        return _ORIG["forward"](self, pair)
    COUNTS["served"] += 1
    if COUNTS["first"] is None:
        COUNTS["first"] = {"mode": "fast", "N": int(x.shape[-2]), "dtype_in": str(x.dtype), "w_dtype": str(self.p_in.weight.dtype), "dir": self.direction,
                           "autocast": torch.is_autocast_enabled(), "out_dtype": str(out.dtype), "word": word}
        print("[fpf_rf3] FIRST CALL", COUNTS["first"], flush=True)
    return out


def enable(mode="fast"):
    """Monkey-patch rf3 TriangleMultiplication.forward class-wide. mode in {'fast','stock'}."""
    global MODE
    import rf3.model.layers.attention as A
    if "forward" not in _ORIG:
        _ORIG["forward"] = A.TriangleMultiplication.forward
    if mode == "stock":
        A.TriangleMultiplication.forward = _ORIG["forward"]
    elif mode == "fast":
        A.TriangleMultiplication.forward = _fast_forward
    else:
        raise ValueError(mode)
    MODE = mode
    COUNTS["served"] = 0; COUNTS["fallback"] = {}; COUNTS["errors"] = 0; COUNTS["first"] = None
    return MODE


def disable():
    return enable("stock")


def describe():
    import fpf_rf3_trimul_rows as TR
    return {"mode": MODE, **COUNTS, "word": TR.STATE["fast"]["word"], "rows": TR.census("fast")}


# =====================================================================================================================================
# Components beyond the trimul: non-trimul pairformer ops.  All class-level monkey patches, no file/weight edits, each independently switchable.
#   triattn:    'stock' | 'gflash' (fused LN+cast(+transpose)+to_b prologue [lnl_fused.ln_linear] + one concatenated q|k|v|g GEMM + the flash_triattn Triton kernel
#                                   instead of cuet.triangle_attention, gate applied in the kernel epilogue [lnl_fused.gate_transpose])
#   transition: 'stock' | 'triton' (the shared core's transition provider opt_core.kernels.transition under the arm's tier word fast | big: its measured row per width / size / card)
#   apb:        'stock' | 'triton' (bias B = to_b(ln_0(Z)) via lnl_fused.ln_linear; rest stock) | 'safe' (stock ops, CUDA-graph capturable)
#   levers:     set_kit_levers() turns the kit's RF3_CUDAGRAPH / RF3_HOIST runtime flags on (only when the kit's patched rf3 files are installed)
# Every non-stock mode is tolerance-class (bf16 operands / fp32 accumulation & LN statistics; accumulation order differs) unless proven bitwise.
# =====================================================================================================================================
CFG2 = {"triattn": "stock", "transition": "stock", "apb": "stock", "levers": None, "warm": False}
# Lever sub-steps of the arm's `@L1` step (`@L1.warm`): the graph roll-out's fixed-cost levers, each switchable alone.
#   warm  -> rf3.graph_flags.set_levers(warm=True): the sampler's eager warm-up runs 3 steps before the process's first capture and 1 afterwards
#            (bitwise by construction: the warm-up touches only static clones re-initialised after capture and consumes no draw).
LEVER_SUBS = ("warm",)
COUNTS2 = {"triattn": {}, "transition": {}, "apb": {}}


def _c2(kind, key):
    COUNTS2[kind][key] = COUNTS2[kind].get(key, 0) + 1


def _cache(mod):
    c = getattr(mod, "_fpf_cache", None)
    if c is None:
        c = mod._fpf_cache = {}
    return c


def set_kit_levers(warm=False):
    try:
        import rf3.graph_flags as GF
    except Exception:
        CFG2["levers"] = "unavailable"; CFG2["warm"] = False; return False
    GF.CUDAGRAPH_MODE = "1"; GF.HOIST = True; GF.GRAPH_SAFE_OPS = True
    CFG2["levers"] = True
    if hasattr(GF, "set_levers"):                       # a graph_flags without the warm-up lever -> named, not silent
        GF.set_levers(warm=bool(warm)); CFG2["warm"] = bool(warm)
    else:
        CFG2["warm"] = "unavailable" if warm else False
    return True


def _lever_step(lv):
    """'L1[.sub...]' -> the sub-steps named (LEVER_SUBS), refused by name otherwise."""
    step, *subs = lv.split(".")
    if step != "L1": raise ValueError("unknown lever step %r" % step)
    bad = [x for x in subs if x not in LEVER_SUBS]
    if bad: raise ValueError("unknown lever sub-step(s) %r of %r (known: %r)" % (bad, lv, LEVER_SUBS))
    return subs


# ---------------- triangle attention
def _tri_kernel():
    import flash_triattn as FT   # noqa: F401  -- routed + imported here (the kit's kernels census lists it); what launches is the provider's row for the arm's
    import fpf_rf3_triattn as R  # 'gflash.<word>' sub-word (opt_core.kernels.triattn; default word flash = FT.flash_triangle_attention, this module object)
    return R.gflash_kernel


def _tri_weights(m):
    c = _cache(m)
    if "tri_glue" not in c:
        dt = torch.bfloat16
        Wcat = torch.cat([m.to_q.weight, m.to_k.weight, m.to_v.weight, m.to_g.weight], 0).detach().to(dt).contiguous()
        nq = m.to_q.weight.shape[0]
        bcat = torch.zeros(Wcat.shape[0], device=Wcat.device, dtype=dt)
        if m.to_g.bias is not None:
            bcat[3 * nq:] = m.to_g.bias.detach().to(dt)
        for lin in (m.to_q, m.to_k, m.to_v):
            assert lin.bias is None
        h = m.to_b.weight.shape[0]; NOUT = 16 if h <= 16 else 1 << (h - 1).bit_length()
        Wb = torch.zeros((NOUT, m.to_b.weight.shape[1]), device=Wcat.device, dtype=dt); Wb[:h] = m.to_b.weight.detach().to(dt)
        assert m.to_b.bias is None
        c["tri_glue"] = dict(Wcat=Wcat, bcat=bcat, nq=nq, Wb=Wb, h=h, lnw=m.norm.weight.detach().float().contiguous(), lnb=m.norm.bias.detach().float().contiguous(), eps=m.norm.eps)
    return c["tri_glue"]


def _tri_forward_v2(self, pair):
    from foundry import SHOULD_USE_CUEQUIVARIANCE
    from einops import rearrange
    mode = CFG2["triattn"]
    if not (self.use_cuequivariance and SHOULD_USE_CUEQUIVARIANCE) or not pair.is_cuda or not torch.is_autocast_enabled():
        _c2("triattn", "fallback:vanilla/no-autocast"); return _ORIG["tri_forward"](self, pair)
    C = pair.shape[-1]
    if C not in (64, 128) or pair.dim() not in (3, 4):
        _c2("triattn", "fallback:C=%d" % C); return _ORIG["tri_forward"](self, pair)
    import lnl_fused as RFU, torch.nn.functional as Fn
    W = _tri_weights(self)
    y16, b16 = RFU.ln_linear(pair, W["lnw"], W["lnb"], W["Wb"], eps=W["eps"], write_y=True, transpose=not self.start_node)   # y16 in the layout the projections consume
    bias = b16[..., :W["h"]]                                     # [b, i, j, h] (untransposed, as stock: bias from norm(pair) before the rearrange)
    qkvg = Fn.linear(y16, W["Wcat"], W["bcat"])                   # [b, i', j', 4*nq]
    nq = W["nq"]
    query = rearrange(qkvg[..., 0:nq], "b i j (h d) -> b i h j d", h=self.h); key = rearrange(qkvg[..., nq:2 * nq], "b i k (h d) -> b i h k d", h=self.h)
    value = rearrange(qkvg[..., 2 * nq:3 * nq], "b i k (h d) -> b i h k d", h=self.h)
    if y16.dim() == 3:   # unbatched call: add batch dim for the kernel like cuEq does implicitly
        query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0); bias_cueq = rearrange(bias, "i j h -> 1 1 h i j")
    else:
        bias_cueq = rearrange(bias, "b i j h -> b 1 h i j")
    kern = _tri_kernel()
    try:
        out = kern(query, key, value, bias=bias_cueq, scale=self.scaling)
    except Exception as e:  # out-of-memory propagates (the contiguous retry needs more memory, not less)
        if is_oom(e): raise
        _c2("triattn", "kernel-error->contiguous:" + repr(e)[:60])
        out = kern(query.contiguous(), key.contiguous(), value.contiguous(), bias=bias_cueq.contiguous(), scale=self.scaling)
    if out.is_contiguous() and out.dtype == torch.bfloat16:
        # fused epilogue: sigmoid(gate) * out (+ transpose back for the ending node) in one Triton pass; to_out then runs on a contiguous tensor
        q4 = qkvg if qkvg.dim() == 4 else qkvg.unsqueeze(0)
        y = RFU.gate_transpose(out, q4.contiguous(), 3 * nq, transpose=not self.start_node)
        if y16.dim() == 3:
            y = y[0]
        _c2("triattn", "served:" + mode + "+gatek"); return self.to_out(y)
    if y16.dim() == 3:
        out = out[0]; out = rearrange(out, "i h j d -> i j (h d)")
    else:
        out = rearrange(out, "b i h j d -> b i j (h d)")
    gate = torch.sigmoid(qkvg[..., 3 * nq:4 * nq])
    out = gate * out
    if not self.start_node:
        out = rearrange(out, "b i j d -> b j i d") if out.dim() == 4 else rearrange(out, "i j d -> j i d")
    _c2("triattn", "served:" + mode); return self.to_out(out)


# ---------------- transition: the shared core's transition PROVIDER under the arm's TIER word (fast | big)
TTR_PROVIDER = "opt_core.kernels.transition"
TTR_SERVE_ROWS = ("v2", "v1", "pf", "lnl")     # the provider rows this arm launches (Triton cells with fixed launch settings, CUDA-graph capturable); any other
                                               # row the provider names for a cell (an autotuning / sealed-package row) steps aside BY NAME to the module's own forward
_TTR_MEMO = {}                                  # (C, HID, I, rows, residual, capturing) -> (row or None, census word)


def _ttr_word():
    """The tier word this arm asks: the arm's head tier (CFG2 fast_word: fast | big), fast for a row-word arm."""
    w = str(CFG2.get("fast_word") or "fast")
    return w if w in ("fast", "big") else "fast"


def _ttr_stack(TR, dev):
    try:
        return TR.stack_word(dev)
    except Exception:
        return None


def _transition_forward_v2(self, X):
    mode = CFG2["transition"]
    C = X.shape[-1]; HID = self.linear_1.weight.shape[0]
    if not X.is_cuda or not torch.is_autocast_enabled():
        _c2("transition", "fallback:cpu/no-autocast"); return _ORIG["tr_forward"](self, X)
    if mode == "triton":
        import sys as _sys
        TR = _sys.modules.get(TTR_PROVIDER) or __import__(TTR_PROVIDER, fromlist=["transition"])
        c = _cache(self)
        W = c.get("tr_W")
        if W is None:
            for lin in (self.linear_1, self.linear_2, self.linear_3):          # the mapper rule: a parameter the provider is not handed must not exist
                if getattr(lin, "bias", None) is not None:
                    _c2("transition", "fallback:C=%d,HID=%d" % (C, HID)); return _ORIG["tr_forward"](self, X)
            W = TR.pack(w_o=self.linear_3.weight, w_a=self.linear_1.weight, w_b=self.linear_2.weight, ln_w=self.layer_norm_1.weight,
                        ln_b=self.layer_norm_1.bias, eps=self.layer_norm_1.eps)
            if _ttr_word() != "big":                                       # fast keeps the packed weights across calls (+~0.2 GiB resident for the model's transitions);
                c["tr_W"] = W                                                # big packs per call and lets the copy go: the memory row holds no duplicate of the module's weights
            if "tr_stack" not in c:
                c["tr_stack"] = _ttr_stack(TR, X.device)
        stack = c["tr_stack"]
        _res = bool(getattr(self, "_fpf_res", False)) and X.dtype == torch.bfloat16
        I = int(X.shape[-2]) if X.dim() >= 2 else 1                           # the track's token count (z: [.., I, I, c]; msa rows: [.., S, I, c]; s: [.., I, c])
        rows = X.numel() // C
        capturing = bool(torch.cuda.is_current_stream_capturing())
        _sq = X.dim() >= 3 and X.shape[-3] == X.shape[-2]                          # the provider's row family from the call's own shape: square token dims = the pair track;
        fam = "single" if (not _sq and TR.cell_word(int(C), int(HID), "single") is not None) else "pair"   # a non-square input of a width it tables as single-track = single; else pair (its 64-wide rows family included)
        memo = (int(C), int(HID), I, int(rows), _res, capturing, fam)
        hit = _TTR_MEMO.get(memo)
        if hit is None:
            try:
                sel = TR.select(_ttr_word(), c=int(C), hidden=int(HID), n_tokens=I, dtype="bf16", direction="fwd", timing="eager", family=fam,
                                residual=_res, cc=None, device=X.device, stack=stack, capture=capturing, rows_count=int(rows), ln_given=False)
                if sel.row in TR.STOCK_ROWS:
                    hit = (None, "fallback:C=%d,HID=%d" % (C, HID))               # the provider's cell names the statements for this width (the 384-wide single transition): the module's own forward, by name
                elif sel.row not in TTR_SERVE_ROWS or (capturing and "warm_before_capture" in (sel.words or ())):
                    hit = (None, "fallback:C=%d,HID=%d:row=%s" % (C, HID, sel.row))   # a row this arm does not launch (autotuning / sealed): the module's own forward, by name
                else:
                    hit = (sel.row, "served:%s:C=%d" % (sel.row, C))
            except TR.Refusal as e:
                kind = str(e).split(" ")[0].replace(",", ";").replace("=", "~")[:72]
                hit = (None, ("fallback:N<=%d:%s" % (I, kind)) if ("below" in kind or "no_cell" in kind and I <= 256) else ("fallback:C=%d,HID=%d:%s" % (C, HID, kind)))
            _TTR_MEMO[memo] = hit
        row, word = hit
        if row is None:
            _c2("transition", word); return _ORIG["tr_forward"](self, X)
        try:
            out, sel = TR.transition(X, W, word=_ttr_word(), residual=_res, n_tokens=I, family=fam, stack=stack, capture=capturing)
        except TR.Refusal as e:                                                # refused at launch: the module's own forward, counted by name
            _c2("transition", "fallback:C=%d,HID=%d:%s" % (C, HID, str(e).split(" ")[0][:72])); return _ORIG["tr_forward"](self, X)
        _c2("transition", word + ("+res" if _res else "")); self._fpf_res_done = _res
        return out
    return _ORIG["tr_forward"](self, X)


# ---------------- attention pair bias (pairformer): only the pair-side LN + to_b is replaced (that is where the time is at large N)
def _sqrt_c(m, Q):
    c = _cache(m); k = ("sqrt_c", Q.device, Q.dtype)
    if k not in c:
        c[k] = torch.sqrt(torch.tensor(m.c).to(Q.device, Q.dtype))   # same op sequence as stock -> same value; computed once
    return c[k]


def _apb_forward_v2(self, A_I, S_I, Z_II, Beta_II=None):
    mode = CFG2["apb"]
    if self.use_deepspeed_evo or not Z_II.is_cuda or not torch.is_autocast_enabled() or S_I is not None:
        _c2("apb", "fallback"); return _ORIG["apb_forward"](self, A_I, S_I, Z_II, Beta_II)
    C = Z_II.shape[-1]
    if mode == "triton":
        if C not in (64, 128) or self.n_head not in (4, 8, 16, 32) or Z_II.dim() not in (3, 4):
            _c2("apb", "fallback:C=%d,h=%d" % (C, self.n_head)); return _ORIG["apb_forward"](self, A_I, S_I, Z_II, Beta_II)
        import lnl_fused as RFU
        c = _cache(self)
        if "apb_w" not in c:
            h = self.n_head; NOUT = 16 if h <= 16 else 32
            Wb = torch.zeros((NOUT, C), device=Z_II.device, dtype=torch.bfloat16); Wb[:h] = self.to_b.weight.detach().to(torch.bfloat16)
            c["apb_w"] = (self.ln_0.weight.detach().float().contiguous(), self.ln_0.bias.detach().float().contiguous(), Wb, self.ln_0.eps)
        ln0w, ln0b, Wb, eps0 = c["apb_w"]
        # ---- stock code with the B_IIH line replaced ----
        A_I = self.ln_1(A_I)
        if self.force_bfloat16 and A_I.device.type != "mps":
            A_I = A_I.to(torch.bfloat16)
        Q_IH = self.to_q(A_I); K_IH = self.to_k(A_I); V_IH = self.to_v(A_I)
        _, b16 = RFU.ln_linear(Z_II, ln0w, ln0b, Wb, eps=eps0, write_y=False)
        import fpf_rf3_apb_rows as APBR                       # the head sub-word apb.<word>: None = the statement below (fpf_rf3_apb_rows)
        core = APBR.core_word()
        if core is None:
            B_IIH = b16[..., :self.n_head] + Beta_II[..., None]
        G_IH = self.to_g(A_I)
        if core is not None:
            A_f = APBR.serve_core(self, Q_IH, K_IH, V_IH, b16, Beta_II, G_IH)
            if A_f is not None:
                _c2("apb", "served:triton+" + core); return self.to_a(A_f)
            B_IIH = b16[..., :self.n_head] + Beta_II[..., None]  # declined by name (the rows census says why): the statement
        Q_IH = Q_IH / _sqrt_c(self, Q_IH)
        A_IIH = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2)
        A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH); A_I = G_IH * A_I; A_I = A_I.flatten(start_dim=-2)
        _c2("apb", "served:triton"); return self.to_a(A_I)
    if mode == "safe":   # stock ops byte-for-byte, except the per-call torch.tensor(c).to(device) H2D copy is replaced by a cached identical tensor (CUDA-graph capturable)
        A_I = self.ln_1(A_I)
        if self.force_bfloat16 and A_I.device.type != "mps":
            A_I = A_I.to(torch.bfloat16)
        Q_IH = self.to_q(A_I); K_IH = self.to_k(A_I); V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II)) + Beta_II[..., None]
        G_IH = self.to_g(A_I)
        Q_IH = Q_IH / _sqrt_c(self, Q_IH)
        A_IIH = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2)
        A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH); A_I = G_IH * A_I; A_I = A_I.flatten(start_dim=-2)
        _c2("apb", "served:safe"); return self.to_a(A_I)
    return _ORIG["apb_forward"](self, A_I, S_I, Z_II, Beta_II)


def enable_v2(triattn=None, transition=None, apb=None):
    """Set the component modes (None = leave unchanged). Installs/uninstalls the class-level patches accordingly."""
    import rf3.model.layers.attention as A, rf3.model.layers.layer_utils as LU, rf3.model.layers.pairformer_layers as PL
    if "tri_forward" not in _ORIG:
        _ORIG["tri_forward"] = A.TriangleAttention.forward; _ORIG["tr_forward"] = LU.Transition.forward; _ORIG["apb_forward"] = PL.AttentionPairBiasPairformerDeepspeed.forward
    if triattn is not None: CFG2["triattn"] = triattn
    if transition is not None: CFG2["transition"] = transition
    if apb is not None: CFG2["apb"] = apb
    A.TriangleAttention.forward = _ORIG["tri_forward"] if CFG2["triattn"] == "stock" else _tri_forward_v2
    LU.Transition.forward = _ORIG["tr_forward"] if CFG2["transition"] == "stock" else _transition_forward_v2
    PL.AttentionPairBiasPairformerDeepspeed.forward = _ORIG["apb_forward"] if CFG2["apb"] == "stock" else _apb_forward_v2
    for k in COUNTS2: COUNTS2[k] = {}
    return dict(CFG2)


def describe_v2():
    return {"cfg": dict(CFG2), "counts": {k: dict(v) for k, v in COUNTS2.items()}}


def apply_arm(arm):
    """Composite arm string: '<trimul>[+gflash][+ttr][+apb|+sapb][@L1[.warm]]' (the re-definitions further down extend the grammar with
    'tg', 'dattn', 'res', 'xatt', 'xmul', 'msa', 'smsa', 'xln' and the '.<word>' sub-words).
    trimul in {stock, fast}; e.g. 'stock', 'fast', 'fast+gflash+ttr+apb@L1'; '@L1' turns the kit's runtime flags on (set_kit_levers),
    its sub-steps (LEVER_SUBS) the roll-out's fixed-cost levers."""
    lever = None; subs = []
    if "@" in arm:
        arm, lv = arm.split("@", 1)
        subs = _lever_step(lv)
        lever = True
    parts = [p for p in arm.split("+") if p]
    trimul = "stock"; tri = "stock"; tr = "stock"; apb = "stock"
    for p in parts:
        if p in ("stock", "fast"): trimul = p
        elif p == "gflash": tri = p
        elif p == "ttr": tr = "triton"
        elif p == "apb": apb = "triton"
        elif p == "sapb": apb = "safe"
        else: raise ValueError("unknown arm component %r" % p)
    enable(trimul); enable_v2(triattn=tri, transition=tr, apb=apb)
    if lever is not None:
        set_kit_levers(warm="warm" in subs)
    else:
        CFG2["warm"] = False
    return {"trimul": trimul, "triattn": tri, "transition": tr, "apb": apb, "levers": CFG2["levers"], "warm": CFG2["warm"]}


# =====================================================================================================================================
# Trunk graph (arm component 'tg'): CUDA-graph capture of the 48-block pairformer stack.  The stack is called once per recycle with
# static shapes (S_I [I,384], Z_II [I,I,128]); at small N it is launch-bound (thousands of short kernels per recycle).
# Capture once per (I, dtypes, arm config) after an eager warm-up pass on a side stream; replay = copy inputs into the static buffers,
# graph.replay(), clone outputs.  Template embedder + MSA module stay eager.  Kernels replayed are exactly the ones the eager path launches
# (numerics identical to the same arm run eagerly, given deterministic kernels) -> inherits the tier of the arm's other components.
# =====================================================================================================================================
TG = {"on": False, "graphs": {}, "captures": [], "replays": 0, "fallbacks": {}}
TG_RETRIES = 2           # capture attempts per shape; after that the shape stays eager (counted shape_gave_up), other shapes keep their graphs
TG_WARMUP = 3            # eager passes on a side stream before capture (>= 2: lazy first/second-call work of every kernel path — JIT, fast-launch checks,
                         # numerics probes — must happen OUTSIDE capture)


def _block_forward_graphsafe(block, S_I, Z_II, beta):
    Z_II = Z_II + block.drop_row(block.maybe_make_batched(block.tri_mul_outgoing)(Z_II))
    Z_II = Z_II + block.drop_row(block.maybe_make_batched(block.tri_mul_incoming)(Z_II))
    Z_II = Z_II + block.drop_row(block.maybe_make_batched(block.tri_attn_start)(Z_II))
    Z_II = Z_II + block.drop_col(block.maybe_make_batched(block.tri_attn_end)(Z_II))
    Z_II = Z_II + block.z_transition(Z_II)
    if S_I is not None:
        S_I = S_I + block.attention_pair_bias(S_I, None, Z_II, Beta_II=beta)
        S_I = S_I + block.s_transition(S_I)
    return S_I, Z_II


def _stack_eager(rec, S_I, Z_II, beta):
    for block in rec.pairformer_stack:
        S_I, Z_II = _block_forward_graphsafe(block, S_I, Z_II, beta)
    return S_I, Z_II


def _recycler_forward_tg(self, f, S_inputs_I, S_init_I, Z_init_II, S_I, Z_II):
    Z_II = Z_init_II + self.process_zh(Z_II)
    Z_II = Z_II + self.template_embedder(f, Z_II)
    Z_II = self.msa_module(f, Z_II, S_inputs_I)
    S_I = S_init_I + self.process_sh(S_I)
    if not TG["on"] or torch.is_grad_enabled() or not Z_II.is_cuda:
        TG["fallbacks"]["grad_or_off"] = TG["fallbacks"].get("grad_or_off", 0) + 1
        for block in self.pairformer_stack:
            S_I, Z_II = block(S_I, Z_II)
        return S_I, Z_II
    key = (tuple(S_I.shape), tuple(Z_II.shape), S_I.dtype, Z_II.dtype, MODE, CFG2["triattn"], CFG2["transition"], CFG2["apb"], torch.is_autocast_enabled(), bool(globals().get("RES", {}).get("on")), bool(globals().get("DATTN", {}).get("on")))
    ent = TG["graphs"].get(key)
    if ent is None and TG.setdefault("failed", {}).get(key, 0) >= TG_RETRIES:
        TG["fallbacks"]["shape_gave_up"] = TG["fallbacks"].get("shape_gave_up", 0) + 1      # this shape only; other shapes keep their graphs
        for block in self.pairformer_stack:
            S_I, Z_II = block(S_I, Z_II)
        return S_I, Z_II
    if ent is None:
        if len(TG["graphs"]) >= int(os.environ.get("FPF_RF3_TG_MAX", "6")):   # bound graph-pool memory: evict the least-recently used graph
            lru = min(TG["graphs"].items(), key=lambda kv: kv[1].get("last", 0.0))[0]
            del TG["graphs"][lru]; torch.cuda.synchronize(); torch.cuda.empty_cache()
            TG["fallbacks"]["evictions"] = TG["fallbacks"].get("evictions", 0) + 1
        t0 = time.perf_counter()
        beta = torch.zeros(1, device=Z_II.device)
        sS = S_I.clone(); sZ = Z_II.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(TG_WARMUP):
                _stack_eager(self, sS, sZ, beta)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                oS, oZ = _stack_eager(self, sS, sZ, beta)
            torch.cuda.synchronize()
        except Exception as e:  # out-of-memory during capture propagates; any other capture failure -> eager for this call, counted
            if is_oom(e): raise
            TG["failed"][key] = TG["failed"].get(key, 0) + 1
            print("[fpf_rf3 tg] CAPTURE FAILED for I=%d (attempt %d; eager for this call, graphs of other shapes kept): %s" % (Z_II.shape[-2], TG["failed"][key], repr(e)[:300]), flush=True)
            TG["fallbacks"]["capture_failed"] = TG["fallbacks"].get("capture_failed", 0) + 1
            try:
                del g
            except Exception:
                pass
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            for block in self.pairformer_stack:
                S_I, Z_II = block(S_I, Z_II)
            return S_I, Z_II
        ent = dict(g=g, sS=sS, sZ=sZ, oS=oS, oZ=oZ, beta=beta)
        TG["graphs"][key] = ent
        TG["captures"].append({"I": int(Z_II.shape[-2]), "capture_s": time.perf_counter() - t0, "key": str(key[4:8])})
        print("[fpf_rf3 tg] captured pairformer stack for I=%d in %.2f s (arm %s)" % (Z_II.shape[-2], time.perf_counter() - t0, str(key[4:8])), flush=True)
    ent["sS"].copy_(S_I); ent["sZ"].copy_(Z_II)
    ent["last"] = time.perf_counter(); ent["g"].replay(); TG["replays"] += 1
    return ent["oS"].clone(), ent["oZ"].clone()


def enable_trunk_graph(on):
    import rf3.model.RF3_structure as RS
    if "rec_forward" not in _ORIG:
        _ORIG["rec_forward"] = RS.Recycler.forward
    TG["on"] = bool(on)
    RS.Recycler.forward = _recycler_forward_tg if on else _ORIG["rec_forward"]
    if not on:
        TG["graphs"].clear()
    return TG["on"]


def describe_tg():
    return {"on": TG["on"], "n_graphs": len(TG["graphs"]), "captures": TG["captures"][-6:], "replays": TG["replays"], "fallbacks": dict(TG["fallbacks"])}


_apply_arm_base = apply_arm
def apply_arm(arm):   # noqa: F811  -- extends the component parser with 'tg'
    parts = arm.split("@")[0].split("+")
    tg = "tg" in parts
    arm2 = "+".join(p for p in parts if p != "tg") + (("@" + arm.split("@")[1]) if "@" in arm else "")
    if not arm2.split("@")[0]:
        arm2 = "stock" + arm2
    cfg = _apply_arm_base(arm2)
    enable_trunk_graph(tg)
    cfg["trunk_graph"] = tg
    return cfg


_describe_base = describe_v2
def describe_v2():   # noqa: F811
    d = _describe_base(); d["trunk_graph"] = describe_tg(); d["cfg"]["trunk_graph"] = TG["on"]; return d


# =====================================================================================================================================
# Diffusion attention (arm component 'dattn'): diffusion-transformer attention with pair bias: the 24-block token transformer computes, per block and per step,
#   softmax_j( (Q/sqrt(c)) K^T + B ) V  with eager ops (bf16 QK^T GEMM with h as the LAST logits dim -> strided fp32 softmax -> PV GEMM).
# Replaced by ONE torch.scaled_dot_product_attention call (fused kernel, fp32 softmax/accumulation, bf16 operands: same numerics class, not bitwise).
# Installed by re-compiling the class's own forward source with the 10-line attention block swapped (works on upstream 4010e3e AND on the
# kit's HOIST-patched file, whose surrounding code differs) — nothing else in the method changes; atom-transformer path (Beta_II given) untouched.
# =====================================================================================================================================
_DATTN_BLOCK = '''            Q_IH = Q_IH / np.sqrt(self.c)
            A_IIH = torch.softmax(
                torch.einsum("...ihd,...jhd->...ijh", Q_IH, K_IH) + B_IIH, dim=-2
            )  # softmax over j
            ## G_IH: [B, I, H, C]
            ## A_IIH: [B, I, I, H]
            ## V_IH: [B, I, H, C]
            A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V_IH)
            A_I = G_IH * A_I  # [B, I, H, C]
            A_I = A_I.flatten(start_dim=-2)  # [B, I, Ca]
'''
_DATTN_NEW = '''            A_I = _fpf_dattn(Q_IH, K_IH, V_IH, B_IIH, G_IH, self.c)  # [fpf_rf3 dattn] fused SDPA with pair bias
'''
DATTN = {"on": False, "calls": 0, "fallback": 0}
DATTN_MIN_I = 400        # below this token count the stock math runs (the fused kernel is not faster there), counted fallback


def _fpf_dattn(Q, K, V, B, G, c):
    import math
    Fn = torch.nn.functional
    if Q.shape[-3] < DATTN_MIN_I:                                                          # stock math, bitwise identical to stock
        Qs = Q / np.sqrt(c)
        A_IIH = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Qs, K) + B, dim=-2)
        A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V)
        DATTN["fallback"] += 1
        return (G * A_I).flatten(start_dim=-2)
    q = Q.transpose(-3, -2); k = K.transpose(-3, -2); v = V.transpose(-3, -2)          # [..., H, I, D] views of [..., I, H, D]
    m = B.transpose(-1, -3).transpose(-1, -2)                                          # [..., I, J, H] -> [..., H, I, J]
    squeeze = False
    if q.dim() == 3:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0); squeeze = True
    while m.dim() < q.dim():
        m = m.unsqueeze(0)
    if m.dtype != q.dtype:
        m = m.to(q.dtype)
    o = Fn.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=1.0 / math.sqrt(c))   # [Bt, H, I, D]
    if squeeze:
        o = o[0]
    o = o.transpose(-3, -2)                                                            # [..., I, H, D]
    DATTN["calls"] += 1
    return (G * o).flatten(start_dim=-2)


def enable_dattn(on):
    import inspect, textwrap
    import rf3.model.layers.af3_diffusion_transformer as DT
    from foundry.training.checkpoint import activation_checkpointing
    cls = DT.AttentionPairBiasDiffusion
    if "dattn_forward" not in _ORIG:
        _ORIG["dattn_forward"] = cls.forward
    if not on:
        cls.forward = _ORIG["dattn_forward"]; DATTN["on"] = False; return False
    if "dattn_new" not in _ORIG:
        f = _ORIG["dattn_forward"]
        inner = getattr(f, "__wrapped__", None)
        if inner is None and getattr(f, "__closure__", None):
            for cell in f.__closure__:
                try:
                    if inspect.isfunction(cell.cell_contents) and cell.cell_contents.__name__ == "forward":
                        inner = cell.cell_contents; break
                except ValueError:
                    pass
        inner = inner or f
        src = textwrap.dedent(inspect.getsource(inner))
        src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("@"))      # drop decorator line(s)
        blk = textwrap.dedent(_DATTN_BLOCK); new = textwrap.dedent(_DATTN_NEW)
        # the method body inside the class is indented by 8 in the file; after dedent of the function source the block is indented by 8 as well
        blk8 = "\n".join(("        " + l if l else l) for l in blk.splitlines()) + "\n"; new8 = "\n".join(("        " + l if l else l) for l in new.splitlines()) + "\n"
        if blk8 not in src:
            raise RuntimeError("dattn: attention block text not found in %s.forward source (file %s)" % (cls.__name__, inspect.getsourcefile(inner)))
        src = src.replace(blk8, new8)
        ns = dict(DT.__dict__); ns["_fpf_dattn"] = _fpf_dattn
        exec(compile(src, "<fpf_rf3_dattn:%s>" % inspect.getsourcefile(inner), "exec"), ns)
        _ORIG["dattn_new"] = activation_checkpointing(ns["forward"])
    cls.forward = _ORIG["dattn_new"]; DATTN["on"] = True
    return True


_apply_arm_with_tg = apply_arm
def apply_arm(arm):   # noqa: F811  -- adds 'dattn'
    parts = arm.split("@")[0].split("+")
    da = "dattn" in parts
    arm2 = "+".join(p for p in parts if p != "dattn") + (("@" + arm.split("@")[1]) if "@" in arm else "")
    if not arm2.split("@")[0]:
        arm2 = "stock" + arm2
    cfg = _apply_arm_with_tg(arm2)
    enable_dattn(da); cfg["dattn"] = da
    return cfg


_describe_with_tg = describe_v2
def describe_v2():   # noqa: F811
    d = _describe_with_tg(); d["dattn"] = dict(DATTN); d["cfg"]["dattn"] = DATTN["on"]; return d



# =====================================================================================================================================
# Fused residual adds (arm component 'res'): PairformerBlock.forward is replaced by an equivalent that lets the FPF trimul kernel (K3 epilogue) and the
# fused transition kernel write Z + update directly (3 of the 5 bf16 [N,N,128] residual-add passes per block disappear). Same values and the same
# add semantics as torch (fp32 add of bf16 operands -> bf16); eval mode only (dropout must be a no-op; asserted). Used by both the eager and 'tg' paths.
# =====================================================================================================================================
RES = {"on": False, "fused": 0, "unfused": 0}


def _residual_call(m, Z, batched):
    m._fpf_res = True; m._fpf_res_done = False
    try:
        out = batched(m)(Z) if batched is not None else m(Z)
    finally:
        m._fpf_res = False
    if getattr(m, "_fpf_res_done", False):
        RES["fused"] += 1; return out
    RES["unfused"] += 1; return Z + out


def _pf_block_forward_res(self, S_I, Z_II, beta=None):
    assert not self.training, "fused residual path is inference-only (dropout must be a no-op)"
    mb = self.maybe_make_batched
    Z_II = _residual_call(self.tri_mul_outgoing, Z_II, mb)
    Z_II = _residual_call(self.tri_mul_incoming, Z_II, mb)
    Z_II = Z_II + mb(self.tri_attn_start)(Z_II)
    Z_II = Z_II + mb(self.tri_attn_end)(Z_II)
    Z_II = _residual_call(self.z_transition, Z_II, None)
    if S_I is not None:
        if beta is None:
            c = _cache(self); k = ("beta", Z_II.device)
            if k not in c: c[k] = torch.tensor([0.0], device=Z_II.device)
            beta = c[k]
        S_I = S_I + self.attention_pair_bias(S_I, None, Z_II, Beta_II=beta)
        S_I = S_I + self.s_transition(S_I)
    return S_I, Z_II


def enable_res(on):
    import rf3.model.layers.pairformer_layers as PL
    if "pf_forward" not in _ORIG:
        _ORIG["pf_forward"] = PL.PairformerBlock.forward
    RES["on"] = bool(on); RES["fused"] = 0; RES["unfused"] = 0
    PL.PairformerBlock.forward = _pf_block_forward_res if on else _ORIG["pf_forward"]
    return RES["on"]


_block_forward_graphsafe_base = _block_forward_graphsafe
def _block_forward_graphsafe(block, S_I, Z_II, beta):   # noqa: F811  -- the trunk-graph path honours 'res'
    if RES["on"]:
        return _pf_block_forward_res(block, S_I, Z_II, beta)
    return _block_forward_graphsafe_base(block, S_I, Z_II, beta)


_apply_arm_with_dattn = apply_arm
def apply_arm(arm):   # noqa: F811  -- adds 'res'
    parts = arm.split("@")[0].split("+")
    rs = "res" in parts
    arm2 = "+".join(p for p in parts if p != "res") + (("@" + arm.split("@")[1]) if "@" in arm else "")
    if not arm2.split("@")[0]:
        arm2 = "stock" + arm2
    enable_res(False)
    cfg = _apply_arm_with_dattn(arm2)
    enable_res(rs); cfg["res"] = rs
    return cfg


_describe_with_dattn = describe_v2
def describe_v2():   # noqa: F811
    d = _describe_with_dattn(); d["res"] = dict(RES); d["cfg"]["res"] = RES["on"]; return d


# =====================================================================================================================================
# Triangle-attention ROWS (arm sub-word 'gflash.<word>', component 'xatt[.<word>]'): the shared core's one provider (opt_core.kernels.triattn)
# serves (a) the gflash attention core -- words flash (default) | k2b | k2 | cuda_sm90a |
# fast -- and (b) for the exact tier the stock TriangleAttention statement's own cuequivariance call ('xatt': word exact = exact_headsplit
# where the core's table names it, else the stock op BY NAME; or a row word).  Binding, refusal -> named fallback, census:
# fpf_rf3_triattn.py (its SELECT / CENSUS lines print the provider's describe()).
# =====================================================================================================================================
XATT = {"on": False, "word": None}


def enable_xatt(word=None, on=True):
    import fpf_rf3_triattn as R
    if not on:
        XATT["on"] = R.disable_xatt(); XATT["word"] = None
        return False
    XATT["word"] = R.enable_xatt(word); XATT["on"] = True
    R.register_exit_census()
    return True


_apply_arm_with_res = apply_arm
def apply_arm(arm):   # noqa: F811  -- adds 'gflash.<word>' and 'xatt[.<word>]'
    import fpf_rf3_triattn as R
    arm2, words = R.strip_components(arm)
    enable_xatt(on=False)
    cfg = _apply_arm_with_res(arm2)
    gw = R.set_gflash_word(words["gflash"])
    R.STATE["gflash"]["on"] = CFG2["triattn"] == "gflash"
    if CFG2["triattn"] == "gflash":
        R.register_exit_census()
    if words["gflash"] is not None and words["gflash"] != R.DEFAULT_WORD["gflash"]:
        cfg["gflash_word"] = gw
    if words["xatt"] is not None:
        enable_xatt(words["xatt"])
        cfg["xatt"] = XATT["word"]
    return cfg


_describe_with_res = describe_v2
def describe_v2():   # noqa: F811
    import fpf_rf3_triattn as R
    d = _describe_with_res()
    c = R.census()
    d["gflash_rows"] = c["gflash"]; d["xatt"] = dict(c["xatt"], on=XATT["on"])
    if XATT["on"]:
        d["cfg"]["xatt"] = XATT["word"]
    if R.STATE["gflash"]["word"] != R.DEFAULT_WORD["gflash"]:      # an arm sub-word (gflash.<word>); the bare component's card row prints in the SELECT / CENSUS lines
        d["cfg"]["gflash_word"] = c["gflash"]["word"]
    return d


# =====================================================================================================================================
# Triangle-multiplication ROWS (head sub-word 'fast.<word>', component 'xmul[.<word>]'): the shared core's one provider (opt_core.kernels.trimul)
# serves (a) the fast tier's fused TriMul -- words v4 (default) | tmk3_fast |
# tmk3_exact | tx_sm90a | fast | big (the tmk3 rows also serve the c=64 template pair track) -- and (b) for the exact tier the stock
# TriangleMultiplication statement's own cuequivariance call ('xmul': word exact = tmk3_exact where the core's table names it for this card (bitwise to the
# library op), else the stock op BY NAME; or a row word).  Binding, refusal -> named fallback, census: fpf_rf3_trimul_rows.py (its
# SELECT / CENSUS lines print the provider's describe()).
# =====================================================================================================================================
XMUL = {"on": False, "word": None}


def enable_xmul(word=None, on=True):
    import fpf_rf3_trimul_rows as TR
    if not on:
        XMUL["on"] = TR.disable_xmul(); XMUL["word"] = None
        return False
    XMUL["word"] = TR.enable_xmul(word); XMUL["on"] = True
    TR.register_exit_census()
    return True


_apply_arm_with_triattn = apply_arm
def apply_arm(arm):   # noqa: F811  -- adds 'fast.<word>' and 'xmul[.<word>]'
    import fpf_rf3_trimul_rows as TR
    arm2, words = TR.strip_components(arm)
    enable_xmul(on=False)
    TR.set_fast_word(words["fast"])                        # the word before the arm applies (enable() resets the trimul census; the row binds at the first call)
    cfg = _apply_arm_with_triattn(arm2)
    TR.STATE["fast"]["on"] = MODE == "fast"
    if MODE == "fast":
        TR.register_exit_census()
    if words["fast"] is not None and words["fast"] != TR.DEFAULT_WORD["fast"]:
        cfg["fast_word"] = TR.STATE["fast"]["word"]
    if words["xmul"] is not None:
        enable_xmul(words["xmul"])
        cfg["xmul"] = XMUL["word"]
    return cfg


_describe_with_triattn = describe_v2
def describe_v2():   # noqa: F811
    import fpf_rf3_trimul_rows as TR
    d = _describe_with_triattn()
    c = TR.census()
    d["trimul_rows"] = c["fast"]; d["xmul"] = dict(c["xmul"], on=XMUL["on"])
    if XMUL["on"]:
        d["cfg"]["xmul"] = XMUL["word"]
    if c["fast"]["word"] != TR.DEFAULT_WORD["fast"]:
        d["cfg"]["fast_word"] = c["fast"]["word"]
    return d


# =====================================================================================================================================
# MSA-module ROWS (components 'msa[.opm|.pwa]' and 'smsa'): the MSA module's outer-product-mean and pair-weighted averaging (torch statements
# otherwise) on the shared core's fused MSA-module cells (opt_core.ops.msa_opm / msa_pwa: fast class; opt_core.ops.msa_pwa2:
# the exact-replica kernels behind 'smsa', served per (I, S) class only after a run-time torch.equal against the stock statements,
# the stock forward BY NAME otherwise).  Binding, gates -> named fallback, census: fpf_rf3_msa_rows.py.
# =====================================================================================================================================
MSA = {"on": False, "units": (), "word": None, "card": {}, "exact": False}


def enable_msa(word=None, on=True):
    """word: a COMPONENT_WORDS['msa'] word (None = the bare component = the card's row); off restores the stock forwards."""
    import fpf_rf3_msa_rows as MR
    if on:
        MSA["on"] = MR.enable(on=True, word=word or MR.DEFAULT_WORD["msa"])
    else:
        MSA["on"] = MR.enable(MR.UNITS, False)
    MSA["units"] = tuple(MR.STATE["units"]); MSA["word"] = MR.STATE["word"]; MSA["card"] = dict(MR.STATE["card"])
    return MSA["on"]


def enable_smsa(on=True):
    import fpf_rf3_msa_rows as MR
    MSA["exact"] = MR.enable_exact(on)
    return MSA["exact"]


_apply_arm_with_xmul = apply_arm
def apply_arm(arm):   # noqa: F811  -- adds 'msa[.<word>]' and 'smsa'
    import fpf_rf3_msa_rows as MR
    body, sep, lv = arm.partition("@")
    parts = [p for p in body.split("+") if p]
    word = None; want = False; exact = False; keep = []
    for p in parts:
        head, _, sub = p.partition(".")
        if head == "msa":
            want = True
            word = sub or MR.DEFAULT_WORD["msa"]
            if word not in MR.COMPONENT_WORDS["msa"]:
                raise ValueError("arm %r: unknown msa word %r (words: %s)" % (arm, word, MR.COMPONENT_WORDS["msa"]))
        elif head == "smsa" and not sub:
            exact = True
        else:
            keep.append(p)
    if exact and want and "pwa" in MR.units_for(word)[0]:
        raise ValueError("arm %r: 'smsa' and the fast pair-weighted-averaging cell of 'msa' take the same site — name 'msa.opm+smsa'" % arm)
    arm2 = "+".join(keep) + (sep + lv if sep else "")
    if not arm2.split("@")[0]:
        arm2 = "stock" + arm2
    if MSA["exact"]:
        enable_smsa(False)
    if MSA["on"]:
        enable_msa(on=False)
    cfg = _apply_arm_with_xmul(arm2)
    if want:
        enable_msa(word, True)
    if exact:
        enable_smsa(True)
    cfg.update(_msa_cfg())
    return cfg


def _msa_cfg():
    """The arm record's msa keys: msa = the word applied (False when off), msa_units = the cells bound on this card, msa_aside = a cell the card row leaves at
    stock, by name with its number ({} when none), smsa = the exact pair-weighted-averaging cell on/off."""
    return {"msa": (MSA["word"] or "units") if MSA["on"] else False, "msa_units": list(MSA["units"]) if MSA["on"] else [],
            "msa_aside": dict(MSA["card"].get("aside") or {}) if MSA["on"] else {}, "smsa": bool(MSA["exact"])}


_describe_with_xmul = describe_v2
def describe_v2():   # noqa: F811
    d = _describe_with_xmul()
    if MSA["on"] or MSA["exact"]:
        import fpf_rf3_msa_rows as MR
        d["msa"] = MR.describe()
    else:
        d["msa"] = {"on": False, "served": 0, "exact": {"on": False, "served": 0}}
    d["smsa"] = dict(d["msa"]["exact"])                      # the exact cell's census in its own section (registry probes fpf_v2 smsa served)
    d["cfg"].update(_msa_cfg())
    return d


# LayerNorm ROWS (component 'xln[.<word>]') and the attention-pair-bias head sub-word ('apb.<word>'): the shared core's providers
# opt_core.kernels.ln and opt_core.kernels.apb.  'xln' re-binds torch.nn.LayerNorm.forward: word exact (default) = the
# provider's exact tier per cell (exactln / exactln:widen where its table names them for this card, bitwise to ATen; ATen BY NAME
# at the other cells), fast | big | a row word otherwise; 'apb.<word>' names the pairformer attention core after the fused producer (stmt =
# the module's statements | sdpa | fpf_apb | apb_attn | fast | big, tolerance class, declined by name to the statement when a row refuses).
# Binding, refusal -> named fallback, census: fpf_rf3_ln_rows.py / fpf_rf3_apb_rows.py (SELECT / CENSUS lines print the providers' describe()).
# =====================================================================================================================================
XLN = {"on": False, "word": None}


def enable_xln(word=None, on=True):
    import fpf_rf3_ln_rows as LR
    if not on:
        XLN["on"] = LR.disable_xln(); XLN["word"] = None
        return False
    XLN["word"] = LR.enable_xln(word); XLN["on"] = True
    LR.register_exit_census()
    return True


_apply_arm_with_msa_rows = apply_arm
def apply_arm(arm):   # noqa: F811  -- adds 'xln[.<word>]' and 'apb.<word>'
    import fpf_rf3_ln_rows as LR, fpf_rf3_apb_rows as APBR
    arm2, lw = LR.strip_components(arm)
    arm3, aw = APBR.strip_components(arm2)
    enable_xln(on=False)
    APBR.set_apb_word(aw["apb"])                            # the word before the arm applies (enable_v2 resets the adapter census; the row binds at the first call)
    cfg = _apply_arm_with_msa_rows(arm3)
    APBR.STATE["apb"]["on"] = CFG2["apb"] == "triton"
    if CFG2["apb"] == "triton" and APBR.core_word() is not None:
        APBR.register_exit_census()
        cfg["apb_word"] = APBR.STATE["apb"]["word"]
    if lw["xln"] is not None:
        enable_xln(lw["xln"])
        cfg["xln"] = XLN["word"]
    return cfg


_describe_with_msa_rows = describe_v2
def describe_v2():   # noqa: F811
    import fpf_rf3_ln_rows as LR, fpf_rf3_apb_rows as APBR
    d = _describe_with_msa_rows()
    d["xln"] = dict(LR.census("xln"), on=XLN["on"])
    if XLN["on"]:
        d["cfg"]["xln"] = XLN["word"]
    a = APBR.census("apb")
    d["apb_rows"] = a
    if APBR.core_word() is not None:
        d["cfg"]["apb_word"] = a["word"]
    return d

