"""fpf_rf3_msa_rows.py — the RF3 MSA module's two stock-statement ops on fused Triton cells (arm components 'msa' and 'smsa').

RF3's MSAModule (rf3.model.layers.pairformer_layers: 4 blocks x 10 recycles per item) runs, per block, OuterProductMean_AF3 (c_m=64 -> 32x32 outer
product over the MSA depth -> c_z=128) and MSAPairWeightedAverage (8 heads x 32 channels, per-channel gate) as torch statements in every mode
(einsum -> cuBLAS bmm with the [S, I, I, 1024] / [I, I, 8] intermediates in HBM); its pair track (trimul / triangle attention / transition) is
already the arm's. The cells here are the shared core's fused MSA-module kernels (opt_core.ops.msa_opm, opt_core.ops.msa_pwa: fast class;
opt_core.ops.msa_pwa2: the exact-replica kernels) bound to RF3's module attribute names and statement order:

  component 'msa[.<word>]' (fast class; units opm, pwa — the bare component binds the running card's row CARD_ROWS (9.0: both cells, 8.0: pwa,
  the opm cell's 8.0 tile row being slower than the stock GEMMs there), 'msa.pwa' / 'msa.opm' / 'msa.both' name the cells directly):
    opm   Z += proj_out(einsum(left, right/S)): the stock LayerNorm + proj_left / proj_right / (1/S) statements (bitwise prologue), then ONE
          Triton kernel (msa_opm.opm_core, form 'scalar_norm' epilogue class: outer tile rounded to bf16 where RF3's autocast einsum rounds, the
          K=1024 projection + bias with fp32 accumulation) — the [I, I, 1024] bf16 outer product never reaches HBM.
    pwa   msa += to_out(gate * einsum(softmax_j(bias), v)): the stock norm_pair / to_bias / softmax(dim=-2) statements (bitwise), the fused
          LN -> to_v | to_gate -> sigmoid prologue (msa_pwa.ln_proj_vg), then the per-head contraction with the gate and the K=256 to_out
          projection in the epilogue (msa_pwa.pwa_core_fo) — no [S, I, 256] weights / gated tensors, one store of the update.
  component 'smsa' (exact class candidate; unit pwa2x): the pair-weighted averaging with RF3's own torch statements on the pair side and the
          core's exact-replica kernels on the MSA side (msa_pwa2: _vg / _wv / _po), the K = I contraction issued in a cuBLAS summation
          structure (kmode, R). NOTHING is bitwise by assumption: at a (I, S) class's first call the stock statements run once and every distinct
          candidate is compared torch.equal; the equal one is LOCKED for the process, no equal candidate = the class is served by the stock forward
          BY NAME ('bitcmp_failed:<I>x<S>'), counted. Below the size floor (S < 16 or S*I < 65,536 rows) the stock statements serve by name.

Gates (a call the cell cannot serve runs the class's own forward, counted 'fallback:<reason>'): CUDA; bf16 autocast active (the kernels' operand
class); the served dims (c_m 64, 32x32 outer product -> 128; 8 heads x 32, per-channel gate); rank ([B,S,I,c] / [S,I,c] + [I,I,c]); token floor
MIN_I per unit (FPF_RF3_MSA_OPM_MIN_I / FPF_RF3_MSA_PWA_MIN_I, default 0 = every size).
An out-of-memory inside a served call propagates (opt_core.oom.is_oom): never a fallback.
"""
import os
import types

import torch

from opt_core.oom import is_oom

UNITS = ("opm", "pwa")
COMPONENT_WORDS = {                                       # component 'msa[.<word>]': the word names which fast cells bind; modes.FPF_SUBWORDS restates it (tests/test_modes reads it back)
    "msa": ("card", "pwa", "opm", "both"),               # card (the bare component): the running GPU's row below; pwa / opm: that cell alone; both: the two cells
}
DEFAULT_WORD = {c: w[0] for c, w in COMPONENT_WORDS.items()}
CARD_ROWS = {                                             # the cells the bare component binds, per compute capability of the running device --
    "msa": {(9, 0): "both",                              # 9.0: both cells (the [I,I,1024] intermediate gone);
            (8, 0): "pwa"},                              # 8.0: the pair-weighted-averaging cell alone: the OPM cell's cc-8.0 tile row is slower than
}                                                         # the stock GEMMs at RF3's MSA depth, so the opm cell steps aside BY NAME on that card
CARD_ELSE = {"msa": "pwa"}                                # a device without a row: the pair-weighted-averaging cell alone
CARD_ASIDE = {(8, 0): {"opm": "cc8.0:opm_cell_row_a2=x0.53-0.55_of_cuBLAS"}}   # the census names the cell a card row leaves at stock, with its reason
MIN_I = {"opm": int(os.environ.get("FPF_RF3_MSA_OPM_MIN_I", "0") or 0), "pwa": int(os.environ.get("FPF_RF3_MSA_PWA_MIN_I", "0") or 0)}
STATE = {"on": False, "units": (), "word": None, "card": {}, "counts": {u: {"served": 0, "fallback": {}} for u in UNITS}, "shapes": {}, "cfg": {}, "errors": 0, "first": None}
EXACT = {"on": False, "served": 0, "stock": 0, "fallback": {}, "classes": {}, "probes": []}     # smsa: per (I, S) class -> (kmode, R) locked | None (refused by name)
EXACT_MIN_S, EXACT_MIN_ROWS = 16, 65536                                                              # the exact cell's size floor: below it the stock statements serve by name
_ORIG = {}


def _count(unit, reason):
    d = STATE["counts"][unit]["fallback"]
    d[reason] = d.get(reason, 0) + 1


def _classes():
    import rf3.model.layers.pairformer_layers as PL          # the MSAModule's own namespace: the classes it instantiates
    return PL.OuterProductMean_AF3, PL.MSAPairWeightedAverage


def _autocast_bf16():
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16
    except TypeError:                                        # older signature
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16


# ------------------------------------------------------------------------------------------------------------------ opm (fast)
def _opm_gate(m, msa):
    if msa.dim() != 4:
        return "rank"
    if not msa.is_cuda:
        return "device"
    if not _autocast_bf16():
        return "autocast"
    CH = m.proj_left.out_features
    if CH != 32 or m.proj_right.out_features != CH or m.proj_out.in_features != CH * CH or m.proj_out.out_features != 128:
        return "dims"
    if msa.shape[2] < MIN_I["opm"]:
        return "I<%d" % MIN_I["opm"]
    return None


def _opm_shim(m):
    """msa_opm.pack's 'scalar_norm' form reads the attribute names linear_1 / linear_2 / linear_out / layer_norm; RF3's are proj_left / proj_right /
    proj_out / norm. One namespace per module holds the packed weights (its _fpf_cache), re-packed when a weight changes."""
    key = tuple((p.data_ptr(), p._version) for p in (m.proj_left.weight, m.proj_right.weight, m.proj_out.weight))
    sh = getattr(m, "_rf3msa_opm_shim", None)
    if sh is None or sh.key != key:
        sh = types.SimpleNamespace(linear_1=m.proj_left, linear_2=m.proj_right, linear_out=m.proj_out, layer_norm=m.norm, key=key)
        m._rf3msa_opm_shim = sh
    return sh


def opm_cfg(device):
    """The launch row: msa_opm's ("mask_norm", 128) row for this card (f1 TMA on 9.0, a2 pointer layout on 8.0, p8 without the descriptor API;
    FPF_OPM_CFG=<CFG_VARIANTS name> overrides) — RF3's c_z = 128 cell has that row's geometry (32 x 32 -> 128); the epilogue class is chosen per call."""
    from opt_core.ops import msa_opm as O
    cfg, src = O.cfg_for("mask_norm", 128, device)
    STATE["cfg"]["opm"] = (O.cfg_name(cfg) or "custom") + ":" + src
    return cfg


def _opm_forward(self, msa):
    why = _opm_gate(self, msa)
    if why is not None:
        _count("opm", why)
        return _ORIG["opm"](self, msa)
    try:
        from opt_core.ops import msa_opm as O
        B, N, L = msa.shape[:3]
        x = self.norm(msa)                                   # stock statements (bitwise prologue): LN, the two projections (+bias), right / S
        left = self.proj_left(x)
        right = self.proj_right(x)
        right = right / float(N)
        cache = O.pack(_opm_shim(self), "scalar_norm")
        cfg = opm_cfg(msa.device)
        outs = [O.opm_core(left[b].contiguous().to(torch.bfloat16), right[b].contiguous().to(torch.bfloat16), cache, "scalar_norm",
                           norm_scalar=1.0, proj_in_kernel=True, cfg=cfg) for b in range(B)]      # epilogue EPI 0: proj_out(bf16(outer)) + bias, / 1.0 (exact)
        out = outs[0].unsqueeze(0) if B == 1 else torch.stack(outs, 0)
    except Exception as e:                                   # noqa: BLE001
        if is_oom(e):
            raise
        STATE["errors"] += 1
        if STATE["first"] is None:
            STATE["first"] = "opm: %s: %s" % (type(e).__name__, str(e)[:300])
        raise
    STATE["counts"]["opm"]["served"] += 1
    k = "opm:S%d_I%d" % (N, L)
    STATE["shapes"][k] = STATE["shapes"].get(k, 0) + 1
    return out


# ------------------------------------------------------------------------------------------------------------------ pwa (fast)
def _pwa_gate(m, msa, pair):
    if msa.dim() != 3 or pair.dim() != 3:
        return "rank"
    if not (msa.is_cuda and pair.is_cuda):
        return "device"
    if not _autocast_bf16():
        return "autocast"
    if not m.separate_gate_for_every_channel:
        return "gate_per_head"
    if m.n_heads != 8 or m.weighted_average_channels != 32 or m.msa_channels != 64:
        return "dims"
    if msa.shape[1] < MIN_I["pwa"]:
        return "I<%d" % MIN_I["pwa"]
    return None


def _pwa_wot(m, dtype):
    key = (m.to_out.weight.data_ptr(), m.to_out.weight._version, dtype)
    w = getattr(m, "_rf3msa_pwa_wot", None)
    if w is None or w[0] != key:
        w = (key, m.to_out.weight.detach().to(dtype).t().contiguous())      # [H*C, c_m] (autocast casts the fp32 weight to bf16)
        m._rf3msa_pwa_wot = w
    return w[1]


def _pwa_forward(self, msa_SI, pair_II):
    why = _pwa_gate(self, msa_SI, pair_II)
    if why is not None:
        _count("pwa", why)
        return _ORIG["pwa"](self, msa_SI, pair_II)
    try:
        from opt_core.ops import msa_pwa as P
        S, I = msa_SI.shape[:2]
        H, C = self.n_heads, self.weighted_average_channels
        cfg = P._cfg_override() or P._CFG[C]
        STATE["cfg"]["pwa"] = next((n for n, r in P.CFG_VARIANTS.items() if dict(r) == dict(cfg)), "custom")
        bias_IIH = self.to_bias(self.norm_pair(pair_II))     # stock statements: [I, J, H], softmax over j (dim=-2) in fp32
        w_IIH = torch.softmax(bias_IIH, dim=-2)
        w16 = w_IIH.permute(2, 0, 1).to(torch.bfloat16).contiguous()      # the einsum's own bf16 cast of w, head-major [H, I, J] for the kernel
        m = msa_SI.unsqueeze(0)
        if cfg.get("pro") == "fused" and cfg.get("kernel") != "fg":
            v, g = P.ln_proj_vg(m, self.norm_msa, self.to_v.weight, self.to_gate.weight, sig=not cfg.get("fuse_sig", False),
                               BM=cfg.get("PBM", 64), num_warps=cfg.get("pwarps", 8))
        else:
            mn = self.norm_msa(m)
            v = self.to_v(mn)
            g = self.to_gate(mn)
            if not cfg.get("fuse_sig", False):
                g = torch.sigmoid(g)
            if v.dtype != torch.bfloat16:
                v = v.to(torch.bfloat16); g = g.to(torch.bfloat16)
        v = v.contiguous(); g = g.contiguous()
        if cfg.get("kernel") in ("fo", "fohp"):
            wot = _pwa_wot(self, v.dtype)
            out = torch.empty((S, I, self.msa_channels), dtype=v.dtype, device=v.device)
            P.pwa_core_fo(v[0], w16, g[0], wot, out, H, C, cfg)
        else:                                                # the non-fused-epilogue rows: gated o from the kernel, to_out as the stock GEMM
            o = P.pwa_core(v[0], w16, g[0], H, C, cfg)
            out = self.to_out(o)
    except Exception as e:                                   # noqa: BLE001
        if is_oom(e):
            raise
        STATE["errors"] += 1
        if STATE["first"] is None:
            STATE["first"] = "pwa: %s: %s" % (type(e).__name__, str(e)[:300])
        raise
    STATE["counts"]["pwa"]["served"] += 1
    k = "pwa:S%d_I%d" % (S, I)
    STATE["shapes"][k] = STATE["shapes"].get(k, 0) + 1
    return out


# ------------------------------------------------------------------------------------------------------------------ pwa2x (exact candidate)
def _pwa2_weights(m):
    key = tuple((p.data_ptr(), p._version) for p in (m.to_v.weight, m.to_gate.weight, m.to_out.weight))
    w = getattr(m, "_rf3msa_pwa2_w", None)
    if w is None or w["key"] != key:
        with torch.no_grad():
            w = {"key": key,
                 "wmT": m.to_v.weight.detach().to(torch.bfloat16).t().contiguous(),      # [64, 256] (autocast casts the weight: same bf16 values)
                 "wgT": m.to_gate.weight.detach().to(torch.bfloat16).t().contiguous(),
                 "woT": m.to_out.weight.detach().to(torch.bfloat16).t().contiguous()}    # [256, 64]
        m._rf3msa_pwa2_w = w
    return w


def pwa_exact_candidate(self, msa_SI, pair_II, kmode=0, R=0, BI=64, BS=8, num_warps=4, num_stages=3):
    """MSAPairWeightedAverage.forward with RF3's own pair-side statements (norm_pair, to_bias, softmax over j) and the carried exact-replica kernels on
    the MSA side (msa_pwa2: LN(m) -> RNE bf16 -> v | sigmoid(g); the K = I contraction in summation structure (kmode, R) -> bf16 -> x gate; to_out
    as one K = 256 fp32-accumulated dot, EPI 1 = RF3's single to_out GEMM). Bitwise = stock only when (kmode, R) is the class's compared candidate."""
    import triton
    from opt_core.ops import msa_pwa2 as X
    S, I = msa_SI.shape[:2]
    m = self.norm_msa(msa_SI)                                # stock statement (fp32 LN under autocast)
    W = _pwa2_weights(self)
    x = m.reshape(S * I, X.C_M)
    if not x.is_contiguous():
        x = x.contiguous()
    Mrows = S * I
    V = torch.empty((Mrows, X.HC), device=m.device, dtype=torch.bfloat16); Gt = torch.empty_like(V)
    X._vg_kernel[(triton.cdiv(Mrows, 64),)](x, W["wmT"], W["wgT"], V, Gt, Mrows, C=X.C_M, HCc=X.HC, BM=64, num_warps=8, num_stages=2)
    bias_IIH = self.to_bias(self.norm_pair(pair_II))         # stock statements, byte for byte
    w_IIH = torch.softmax(bias_IIH, dim=-2)
    w16 = w_IIH.permute(2, 0, 1).to(torch.bfloat16).contiguous()          # the einsum's RNE cast of w (values independent of layout), [H, I, J]
    OG = torch.empty((Mrows, X.HC), device=m.device, dtype=torch.bfloat16)
    BK = 16 if kmode == 2 else 64
    X._wv_kernel[(triton.cdiv(I, BI), triton.cdiv(S, BS), X.HEADS)](w16, V, Gt, OG, I, S, int(R), CH=X.C_H, HCc=X.HC, BI=BI, BS=BS, BK=BK,
                                                                       KMODE=2 if kmode == 2 else 0, num_warps=num_warps, num_stages=num_stages)
    del V, Gt
    out = torch.empty((1, S, I, X.C_M), device=m.device, dtype=torch.bfloat16)
    X._po_kernel[(triton.cdiv(Mrows, 128),)](OG, W["woT"], out, Mrows, H=X.HEADS, CH=X.C_H, HCc=X.HC, CM=X.C_M, BM=128, EPI=1, num_warps=4, num_stages=2)
    return out[0]


def pwa_exact_probe(self, msa_SI, pair_II, stock_out=None):
    """Run the stock statements once and every distinct (kmode, R) candidate for I; return {'I','S','stock_ms'?, 'candidates': [(kmode, R, equal, n_diff,
    max_abs)], 'locked': (kmode, R) | None}. The comparison behind the smsa lock."""
    from opt_core.ops import msa_pwa2 as X
    S, I = msa_SI.shape[:2]
    ref = _ORIG.get("pwa", type(self).forward)(self, msa_SI, pair_II) if stock_out is None else stock_out
    rec = {"I": int(I), "S": int(S), "candidates": [], "locked": None}
    for km, R in X.distinct_candidates(int(I)):
        out = pwa_exact_candidate(self, msa_SI, pair_II, kmode=km, R=R)
        eq = bool(torch.equal(out, ref))
        nd = int((out != ref).sum().item())
        mx = float((out.float() - ref.float()).abs().max().item())
        rec["candidates"].append((int(km), int(R), eq, nd, mx))
        if eq and rec["locked"] is None:
            rec["locked"] = (int(km), int(R))
    return rec, ref


def _pwa_exact_gate(m, msa, pair):
    why = _pwa_gate(m, msa, pair)
    if why is not None:
        return why if not why.startswith("I<") else None    # the fast floor is not the exact cell's; its own floor follows
    S, I = msa.shape[:2]
    if S < EXACT_MIN_S or S * I < EXACT_MIN_ROWS:
        return "floor:S<%d_or_rows<%d" % (EXACT_MIN_S, EXACT_MIN_ROWS)
    if msa.dtype not in (torch.bfloat16, torch.float32) or pair.dtype not in (torch.bfloat16, torch.float32):
        return "dtype"
    return None


def _pwa_exact_forward(self, msa_SI, pair_II):
    why = _pwa_exact_gate(self, msa_SI, pair_II)
    if why is not None:
        EXACT["fallback"][why] = EXACT["fallback"].get(why, 0) + 1
        return _ORIG["pwa"](self, msa_SI, pair_II)
    S, I = (int(s) for s in msa_SI.shape[:2])
    cls = (I, S)
    if cls not in EXACT["classes"]:                          # first encounter of the class: the stock statements once + the bit-compare (lock discipline)
        rec, ref = pwa_exact_probe(self, msa_SI, pair_II)
        EXACT["classes"][cls] = rec["locked"]
        EXACT["probes"].append(rec)
        EXACT["stock"] += 1
        return ref                                           # this call is served by the stock statements it just ran (bitwise by construction)
    lock = EXACT["classes"][cls]
    if lock is None:
        w = "bitcmp_failed:%dx%d" % cls
        EXACT["fallback"][w] = EXACT["fallback"].get(w, 0) + 1
        return _ORIG["pwa"](self, msa_SI, pair_II)
    try:
        out = pwa_exact_candidate(self, msa_SI, pair_II, kmode=lock[0], R=lock[1])
    except Exception as e:                                   # noqa: BLE001
        if is_oom(e):
            raise
        STATE["errors"] += 1
        if STATE["first"] is None:
            STATE["first"] = "pwa2x: %s: %s" % (type(e).__name__, str(e)[:300])
        raise
    EXACT["served"] += 1
    return out


# ------------------------------------------------------------------------------------------------------------------ enable / describe
def card_cc():
    """The running device's compute capability (major, minor), or None without CUDA."""
    try:
        return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    except Exception:   # noqa: BLE001
        return None


def units_for(word=None, cc="probe"):
    """The cells a word of component 'msa' binds: card (default) = CARD_ROWS[cc] (CARD_ELSE without a row), pwa | opm = that cell, both = the two.
    Returns (units, card) with card = {"cc": "9.0" | None, "row": <word served>, "aside": {unit: reason}} for the census."""
    word = word or DEFAULT_WORD["msa"]
    if word not in COMPONENT_WORDS["msa"]:
        raise ValueError("msa: unknown word %r (words: %s)" % (word, COMPONENT_WORDS["msa"]))
    if cc == "probe":
        cc = card_cc()
    row = word
    if word == "card":
        row = CARD_ROWS["msa"].get(cc, CARD_ELSE["msa"]) if cc is not None else CARD_ELSE["msa"]
    units = UNITS if row == "both" else (row,)
    aside = {u: r for u, r in CARD_ASIDE.get(cc, {}).items() if u not in units} if word == "card" else {}
    return units, {"cc": ("%d.%d" % cc) if cc else None, "word": word, "row": row, "aside": aside}


def enable(units=UNITS, on=True, word=None):
    """Bind (on) or restore (off) the fast cells class-wide: OuterProductMean_AF3.forward ('opm'), MSAPairWeightedAverage.forward ('pwa'). ``word`` (component
    sub-word) resolves the units by ``units_for`` (the card row for the bare component); ``units`` names them directly otherwise."""
    OPM, PWA = _classes()
    card = {}
    if word is not None:
        units, card = units_for(word)
    if "opm" not in _ORIG:
        _ORIG["opm"] = OPM.forward
    if "pwa" not in _ORIG:
        _ORIG["pwa"] = PWA.forward
    units = tuple(u for u in units if u in UNITS)
    bad = [u for u in units if u not in UNITS]
    if bad:
        raise ValueError("msa: unknown unit(s) %s (units: %s)" % (bad, UNITS))
    OPM.forward = _opm_forward if (on and "opm" in units) else _ORIG["opm"]
    if not EXACT["on"]:
        PWA.forward = _pwa_forward if (on and "pwa" in units) else _ORIG["pwa"]
    elif on and "pwa" in units:
        raise ValueError("msa: the fast pwa cell and the exact pwa2x cell (smsa) take the same site — name one")
    STATE["on"] = bool(on and units); STATE["units"] = units if on else (); STATE["word"] = (word or "units") if on else None; STATE["card"] = card if on else {}
    return STATE["on"]


def enable_exact(on=True):
    """Bind (on) or restore (off) the exact-candidate pair-weighted-averaging cell (component 'smsa') class-wide, under the lock discipline."""
    _OPM, PWA = _classes()
    if "pwa" not in _ORIG:
        _ORIG["pwa"] = PWA.forward
    if on and STATE["on"] and "pwa" in STATE["units"]:
        raise ValueError("smsa: the fast pwa cell (msa) holds the site — name one")
    PWA.forward = _pwa_exact_forward if on else (_pwa_forward if (STATE["on"] and "pwa" in STATE["units"]) else _ORIG["pwa"])
    EXACT["on"] = bool(on)
    return EXACT["on"]


def describe():
    return {"on": STATE["on"], "units": list(STATE["units"]), "word": STATE["word"], "card": dict(STATE["card"]), "min_i": dict(MIN_I), "cfg": dict(STATE["cfg"]),
            "errors": STATE["errors"], "first": STATE["first"], "served": sum(c["served"] for c in STATE["counts"].values()),   # served: the registry's census probe (fpf_v2 msa served)
            "counts": {u: {"served": c["served"], "fallback": dict(c["fallback"])} for u, c in STATE["counts"].items()}, "shapes": dict(STATE["shapes"]),
            "exact": {"on": EXACT["on"], "served": EXACT["served"], "stock": EXACT["stock"], "fallback": dict(EXACT["fallback"]),
                      "classes": {"%dx%d" % k: (list(v) if v else None) for k, v in EXACT["classes"].items()}}}
