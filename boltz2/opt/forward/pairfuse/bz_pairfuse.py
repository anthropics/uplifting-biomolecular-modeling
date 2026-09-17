"""
bz_pairfuse.py — the PAIRFUSE layer driver for Boltz-2 2.2.1 pair stacks (FAST tier; inference only; the boltz wheel is untouched).

What it owns: the number of times the [N, N, 128] pair tensor z crosses HBM per PairformerLayer / PairformerNoSeqLayer.  Stock (and
the per-layer kit-fast path) keep z in fp32 and run, per sub-layer, an fp32 LayerNorm pass, an fp32->bf16 cast pass, the fused core, and an
fp32 residual-add pass; the LayerNorms, casts,
residual adds and the ending node's transposes are roughly half of that traffic.  This driver keeps ONE driver-owned resident copy of z per pair-stack call and
strings the fused cores so that every sub-layer reads z once (LN in the core's prologue, in registers) and writes z once (the residual
added in the core's epilogue, IN PLACE), the ending node addressed by strides, the sequence-attention pair bias projected from the resident
z in one pass:

    site            provider (opt_core)                                                 reads of z   writes of z
    tri_mul_out/in  fpf_trimul_v4.kernels.trimul_v4_forward(z, .., residual=True, out=z)       K1 1 + K3 1  K3 1 (in place)
    tri_att_start   pair_fused.tri_attn_block(z, W, mask, ending=False, residual=True, ln='fused', core=<the triattn provider's row by tier word>)   prologue 1 + epilogue 1   epilogue 1
    tri_att_end     the same with ending=True: prologue reads z^T BY STRIDES, epilogue scatters the update transposed (no transpose pass)
    transition_z    pair_fused.transition(z, T, residual=True, ln='fused', out=z)              1 (+res 1)   1
    seq pair bias   this file's Triton kernel: AttentionPairBias.proj_z (LN + Linear(128->16) + rearrange) in ONE pass over the resident z,
                    fp32 arithmetic on the FMA units (no tensor core, no TF32), installed on the PairformerLayer.attention.proj_z instances so the stock (or
                    mask2's) AttentionPairBias.forward runs unchanged and the layer statement only drops its `z.float()`

Residency (the lever's ONE row word, BOLTZ_PAIRFUSE):
    bf16   z resident in bf16: one owned bf16 buffer per pair-stack call (the caller's fp32 z cast once at entry; every site updates it in
           place: residual added in fp32 registers, stored bf16); fp32 handed back at exit so the modules' external dtype contract is stock's.
           Precision class: the bf16-autocast class of every site's arithmetic (unchanged from kit-fast) PLUS bf16 rounding of the residual
           stream once per site (5 per layer).
    fp32   z resident in fp32 (an owned fp32 buffer): the TriMul sites add their residual in fp32 in K3 (in place); the attention / transition
           sites return their bf16 update and the driver adds it in place (torch add_, 2.5 passes) until a provider takes fp32 res/out
           (no provider row does in this release); LN read from fp32 z in-kernel — pair_fused rows v3_fp32z / v1_fp32x, not admitted here, so this word is
           REFUSED BY NAME at apply (FP32_ROWS_ADMITTED, a source constant: the kit consults no core gate variable).
           Precision class: kit-fast's exactly at every site (same rounding points).
Word grammar: BOLTZ_PAIRFUSE="<residency>[,<site>=<impl>...]" — provider picks per site from PROVIDERS (an ablation of a provider is a
row word, never a toggle of the superseded class lever); the LEVER line credits them: providers=trimul:core.fast,triatt:core.fast,transition:fpf,pairbias:pf
on the fast row, `core.big` at both sites on the memory row (the core providers' TIER words: the measured cell per card and token count names the row).

Hooks (class level, installed by boltz2_opt.pairfuse.apply() at attach time, BEFORE boltz_trunk_levers.apply(): mask2's module-scope
wrappers then wrap these and keep working; resid's layer patches become unreachable for the served stacks and stay in force for the ones
this driver hands back): PairformerModule.forward, PairformerNoSeqModule.forward (pairformer.py), MSALayer.forward (trunkv2.py: its four
stock statements verbatim, the last one — the pair layer — driven here).  PairformerLayer / PairformerNoSeqLayer .forward are NOT patched.
A stack this driver does not serve takes the module's ORIGINAL forward, counted by reason (kernels_off | autocast_off | c:<C> (the template
stack, C=64) | below_min_tokens | training | <a core's refusal word>): never a silent subset.  A kernel error inside a served stack RAISES
(the resident chain cannot be resumed by the stock statements mid-layer): fail-closed, by name.

RNG: every get_dropout_mask draw of the stock statements is KEPT (torch.rand of the stock shape and fp32 dtype, in stock order: 4 per
layer, the 4th column-wise), so the CUDA Philox stream — and therefore every diffusion noise draw after the trunk — is stock's (the
eval-mode mask itself is identically 1.0 at this pin, dropout.py compares `>=` against dropout*training = 0, so dropping the multiply is exact).
Masks: whether pair_mask is trivially all-ones (a single unpadded input: always at the CLI defaults) is decided ONCE per mask TENSOR —
`mask_is_trivial()`: one host sync at the first stack call that sees a given tensor object (identity + version), then answered from a
small cache for every later stack call, layer and recycling pass of the item (the trunk builds pair_mask once per forward); a trivial
mask is handed to the providers as None, a real one in each provider's own layout (numerics identical either way: the providers multiply
/ bias by an all-ones mask exactly).  A caller that replays captured graphs on static buffers sets the predicate per item from the host side
(`with mask_predicate(trivial): ...`) — then no sync is issued at all.

Capture contract (for a graph lever that captures the served modules): after ONE eager call of a shape class (module, N, residency,
device) with a given mask tensor (or under `mask_predicate`), the served path issues NO host synchronisation and takes NO data-dependent
host decision: eligibility is answered from the per-module cache (`_pairfuse_elig`), the trimul provider's warm probe has run for the
dtype class, the pair-bias modules are installed, the mask predicate is cached or given; what remains per layer is fixed-geometry kernel
launches, allocator calls and four `torch.rand` draws (Philox).  `primed(module, z, pair_mask)` answers whether that holds for a call
before it is made; `STATS["host_syncs"]` counts every sync this module issues.  Counting: STATS count BODY executions — a body captured
once and replayed N times by a graph lever is counted here once; the replays are that lever's census (the gate below never compares its
counts with the number of calls the model made).
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
import time
import weakref
from typing import Any, Dict, List, Optional

import torch

TAG = "boltz2-opt"
NAME = "LOCAL.boltz2.pairfuse"      # kit-local strategy id (opt_core report grammar: LOCAL.<kit>.<name>; registry LEVER_IDS)
RESIDENCIES = ("bf16", "fp32")
# The row word: "<residency>[,<site>=<impl>...]" — residency bf16|fp32, then optional provider picks per site.  A provider is a
# callable behind the interface contract v1; the ones named here are what this tree carries (an unknown pick is
# refused by name at apply).
PROVIDERS = {
    "trimul":     {"v4": "opt_core fpf_trimul_v4 (core name) trimul_v4_forward(residual=True, out=z)",
                   "site": "RETIRED (boltz2 0.3.12): the TRIMUL track's in-tree bz2_trimul kit copy is gone — refused by name at apply (trimul:site:kit_copy_retired); the core provider's rows serve its cells: `trimul=core.fast` / `core.v4` / `core.native` (opt_core.kernels.trimul)",
                   "site_tanh": "RETIRED with `site` (the tanh-form sigmoid sub-word of the same kit copy) — refused by name at apply (trimul:site_tanh:kit_copy_retired)"},
    "triatt":     {"k2b": "opt_core.attn.pair_fused tri_attn_block(impl=fpf, ln=fused, core=k2b) -- an available word (the K2B cell by name)",
                   "flash": "the same block with the core's flash_triattn cell as the attention core -- an available word",
                   "core.<word>": "the same block with the attention core served by the core triangle-attention provider (opt_core.kernels.triattn) BY TIER WORD: core.fast (the fast row) / core.big (the memory row) / core.exact -- the provider's measured cell per (cc, dtype, head_dim, heads, N) names the row on each card; a row name (core.k2b, core.triattn_native ...) is an available opt-in word",
                   "default": "the same block with the block's OWN keyed attention core (opt_core.attn.pair_fused core=\"default\", DEFAULT_CORE_TABLE: the triattn provider's `fast` tier word at and above the table's keys threshold per (cc, dtype, head_dim, heads) — flash_triattn by name on its refusal —, the flash_triattn cell below it and for unlisted classes; 0.3.23)"},
    "transition": {"fpf": "opt_core.attn.pair_fused transition(impl=fpf, ln=fused, residual in place)",
                   "core.<word>": "opt_core.kernels.transition transition(word=<word>, residual in place): a provider ROW or TIER word (core.fast = the cell's measured fast winner: v2:fast on 9.0 / v2@<launch> on 8.0 at (128,512); core.v2:fast; core.v1:lnfused ...), opt_core >= 0.5.34.0"},
    "pairbias":   {"pf": "bz_pairfuse.pair_bias (Triton, LN+128->16 fp32 FMA, one pass)"},
}
# `<site>=core.<word>` (sites trimul / triatt): the core's ONE provider for the op (opt_core.kernels.trimul / opt_core.kernels.triattn: every carried
# row + the measured cell table per compute capability) serves the site BY WORD — a row name (triatt: k2b k2 flash cuda_sm90a
# exact_headsplit cueq …; trimul: v4 tmk3_fast tmk3_exact native cueq …) or a tier word (fast | exact | big: the cell's measured winner; the fast / big rows name `trimul=core.fast` in their BOLTZ_PAIRFUSE word).
# A row that cannot serve this box / shape refuses BY NAME before the stack starts (the provider's `Refusal`: the stack's fallback word
# `<site>:<row>:<kind>`, e.g. `triatt:cuda_sm90a:no_prebuilt:…`, `trimul:tx_sm90a:cc:8.0!=9.0(…)`) — nothing is substituted. The picks named in
# PROVIDERS above are this tree's own bindings; the core words are how a measured winner enters a row (modes.py).
CORE_PREFIX = "core."
RETIRED_PICKS = {"trimul": ("site", "site_tanh")}   # picks whose kit copy left this tree: still words of the grammar (parse_word names them), refused BY NAME at apply — nothing is substituted
HANDABLE_ROWS = ("k2b", "flash", "k2", "cuda_sm90a", "triattn_native", "exact_headsplit")   # rows this driver can bind when the provider hands a pick to them by name
PICK_ENV = {"triatt": "BOLTZ_PAIRFUSE_TRIATT", "trimul": "BOLTZ_PAIRFUSE_TRIMUL"}
TIER_PICK_WORDS = ("exact", "fast", "big")             # the provider faces' tier vocabulary: a `core.<tier>` pick is a word of this grammar on every core version (a face that predates a tier word refuses it BY NAME at the probe and the pick steps aside to `fast`, named on the line: handed=)
CORE_SITES = ("trimul", "triatt", "transition")
DEFAULT_PICKS = {"trimul": "v4", "triatt": "core.fast", "transition": "fpf", "pairbias": "pf"}      # the driver's own bindings (the bare driver, the fp32 residency word): the tri-attention site by the core provider's `fast` tier word
RESIDENCY_PICKS = {"bf16": {"triatt": "core.fast"}}   # a residency word's own defaults over DEFAULT_PICKS (the bf16 word: the tri-attention site by the provider's `fast` tier word; the rows name their tier words explicitly: fast `triatt=core.fast`, big `triatt=core.big`)
PF_DEFAULT_CORE = "default"                            # == opt_core.attn.pair_fused.DEFAULT_CORE: the pick `default` hands the block this core word


def default_picks(residency=None) -> dict:
    """The picks a residency word starts from: DEFAULT_PICKS (the driver's own bindings) with the residency's tier words (RESIDENCY_PICKS) over them."""
    d = dict(DEFAULT_PICKS)
    d.update(RESIDENCY_PICKS.get(str(residency or "").strip().lower(), {}))
    return d



def core_word(pick) -> Optional[str]:
    """'core.k2' -> 'k2'; anything else -> None."""
    return pick[len(CORE_PREFIX):] if isinstance(pick, str) and pick.startswith(CORE_PREFIX) and len(pick) > len(CORE_PREFIX) else None


def core_words(site: str) -> tuple:
    """The words the core's provider face for `site` accepts (rows + tiers; the faces are standard-library-only at import)."""
    if site == "triatt":
        from opt_core.kernels import triattn as KA
        return tuple(KA.ROW_NAMES) + tuple(KA.TIER_WORDS)
    if site == "trimul":
        from opt_core.kernels import trimul as KT
        return tuple(KT.ROW_NAMES) + tuple(KT.TIER_WORDS)
    if site == "transition":                                                    # opt_core.kernels.transition: rows v2 / v1 / pf / lnl / ... + tiers
        from opt_core.kernels import transition as KX
        return tuple(KX.ROW_NAMES) + tuple(KX.TIER_WORDS)
    return ()
FP32_ROWS_ADMITTED = False     # the fp32 residency reads LN from fp32 z through pair_fused rows (v3_fp32z / v1_fp32x) that are not admitted in this release: refused by
                                # name at apply (the kit consults no core gate variable; only this source constant admits them)


def parse_word(word: str, environ=None):
    """'bf16' | 'fp32' | 'bf16,trimul=v4,triatt=k2b' -> (residency, picks) or ValueError naming the offending token.  A site's pick may also
    arrive as its own word (PICK_ENV: BOLTZ_PAIRFUSE_TRIATT=core.<row> — the mode rows' form, one lever = one word); the same grammar."""
    toks = [t.strip().lower() for t in str(word).split(",") if t.strip()]
    if not toks or toks[0] not in RESIDENCIES:
        raise ValueError(f"{NAME}: residency word {toks[0] if toks else word!r} not in {RESIDENCIES}")
    env = os.environ if environ is None else environ
    for site, key in PICK_ENV.items():                                # a pick word of its own joins the comma form (the comma form wins when both name the site)
        v = str(env.get(key, "")).strip().lower()
        if v and not any(t.startswith(site + "=") for t in toks[1:]):
            toks.append(f"{site}={v}")
    picks = default_picks(toks[0])                                   # the residency word's own defaults (bf16: triatt=core.fast); the tokens below and the pick words above override them
    for t in toks[1:]:
        if "=" not in t:
            raise ValueError(f"{NAME}: word token {t!r} is not <site>=<impl>")
        site, impl = t.split("=", 1)
        if site not in PROVIDERS:
            raise ValueError(f"{NAME}: unknown site {site!r} (sites: {sorted(PROVIDERS)})")
        if impl not in PROVIDERS[site]:
            w = core_word(impl) if site in CORE_SITES else None
            if w is None or (w.split("@")[0] not in TIER_PICK_WORDS and w.split("@")[0] not in core_words(site)):
                raise ValueError(f"{NAME}: site {site} has no provider {impl!r} in this tree (has: {sorted(PROVIDERS[site])}"
                                 + (f", core.<{'|'.join(core_words(site))}>" if site in CORE_SITES else "") + ")")
        picks[site] = impl
    return toks[0], picks
NMIN = 128                      # below this token count the stack takes the original forward (the class-level block adapters serve it; the ladder starts at 200)
C_SERVED = (128,)

_STATE: Dict[str, Any] = {"applied": False, "residency": None, "picks": {}, "orig": {}, "t_apply": None, "proj_z_patched": [], "word": None, "handed_triatt": {}}
TRIATT_SITE_WORDS = {"cueq": "triatt_cueq", "k2b": "triatt_k2b", "flash": "triatt_flash", "default": "triatt_default", "core": "triatt_core"}   # every word _core_for returns,
SITE_WORDS = ("trimul", "trimul_core") + tuple(TRIATT_SITE_WORDS.values()) + ("transition", "pair_bias", "pair_bias_stock")                            # by the pick it serves: the one
                                                                                                                                                         # table the sites census is seeded from
STATS: Dict[str, Any] = {"stack_calls": 0, "stack_served": 0, "served_by": {}, "fallback": {}, "layers": 0, "noseq_layers": 0,
                         "sites": {w: 0 for w in SITE_WORDS},        # per site word; counted with _count (a word outside the table is tallied, never a KeyError)
                         "core_cells": {}, "handed": {}, "mask_trivial": 0, "mask_real": 0, "mask_pred_synced": 0, "mask_pred_cached": 0, "mask_pred_given": 0, "host_syncs": 0,
                         "errors": {}, "shapes": {}, "alloc": {}, "first_served": None}


def _count(d: dict, k: str, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


# =====================================================================================================================================
# the sequence-attention pair bias: LN(z) @ W^T -> [B, H, N, N] fp32 in ONE pass over the resident z (Triton)
# =====================================================================================================================================
_PB = {"kernel": None}


def _pair_bias_kernel():
    if _PB["kernel"] is not None:
        return _PB["kernel"]
    import triton
    import triton.language as tl

    @triton.jit
    def _pb_kernel(z_ptr, lnw_ptr, lnb_ptr, w_ptr, out_ptr, M, stride_zr, eps,
                   C: tl.constexpr, H: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < M
        cols = tl.arange(0, C)
        r64 = rows.to(tl.int64)
        x = tl.load(z_ptr + r64[:, None] * stride_zr + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)   # [BM, C] (bf16 or fp32 z, cast in registers)
        mean = tl.sum(x, axis=1) / C
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        lnw = tl.load(lnw_ptr + cols).to(tl.float32)
        lnb = tl.load(lnb_ptr + cols).to(tl.float32)
        y = xc * rstd[:, None] * lnw[None, :] + lnb[None, :]                                   # fp32 LayerNorm output (torch: biased variance, rsqrt(var+eps))
        hh = tl.arange(0, H)
        acc = tl.zeros([BM, H], dtype=tl.float32)
        for h in tl.static_range(H):                                                           # the 128->H projection as H broadcast-multiply + row reductions in fp32
            w_h = tl.load(w_ptr + h * C + cols).to(tl.float32)                                 # FMA units, no tensor core: really fp32 (no TF32 anywhere), and 2.4x faster
            col = tl.sum(y * w_h[None, :], axis=1)                                             # than tl.dot(input_precision="ieee") at this [BM,128]x[128,16] shape (pb_bench.py)
            acc = tl.where(hh[None, :] == h, col[:, None], acc)
        optr = out_ptr + hh[None, :].to(tl.int64) * M + r64[:, None]                          # out[h, row]: the 'b i j h -> b h i j' rearrange folded into the store
        tl.store(optr, acc, mask=rmask[:, None])

    _PB["kernel"] = _pb_kernel
    return _pb_kernel


def pair_bias(z: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, w: torch.Tensor, eps: float = 1e-5, BM: int = 64) -> torch.Tensor:
    """AttentionPairBias.proj_z on the resident z: Rearrange('b i j h -> b h i j')(Linear(LayerNorm(z))) -> [B, H, N, N] fp32, one pass over z.
    z [B, N, N, C] bf16|fp32 (last dim contiguous, rows uniformly strided); ln_w/ln_b [C] fp32; w [H, C] fp32 (no bias — boltz's proj_z Linear has none)."""
    import triton
    assert z.dim() == 4 and z.stride(-1) == 1
    B, N, N2, C = z.shape
    H = int(w.shape[0])
    out = torch.empty((B, H, N, N2), dtype=torch.float32, device=z.device)
    k = _pair_bias_kernel()
    M = N * N2
    w32 = w if (w.dtype == torch.float32 and w.is_contiguous()) else w.float().contiguous()
    lw = ln_w if ln_w.dtype == torch.float32 else ln_w.float()
    lb = ln_b if ln_b.dtype == torch.float32 else ln_b.float()
    for b in range(B):
        zb = z[b]
        if not (zb.stride(1) == C and zb.stride(0) == N2 * C):
            zb = zb.contiguous()
        grid = (triton.cdiv(M, BM),)
        k[grid](zb, lw, lb, w32, out[b], M, C, float(eps), C=C, H=H, BM=BM, num_warps=4)
    return out


class FusedProjZ(torch.nn.Module):
    """Stands in for AttentionPairBias.proj_z = Sequential(LayerNorm(c_z), Linear(c_z, H, bias=False), Rearrange('b ... h -> b h ...')) on ONE
    instance: the same function of z in one Triton pass (fp32 arithmetic) for a CUDA z of C in {64,128,256} and H a power of two <= 64;
    anything else (or a z that is not 4-D) takes the original Sequential on z.float() — counted."""

    def __init__(self, orig: torch.nn.Sequential):
        super().__init__()
        self.orig = orig
        ln, lin = orig[0], orig[1]
        self.eps = float(ln.eps)
        self.ok = (isinstance(ln, torch.nn.LayerNorm) and isinstance(lin, torch.nn.Linear) and lin.bias is None and ln.weight is not None
                   and int(lin.weight.shape[1]) in (64, 128, 256) and int(lin.weight.shape[0]) in (4, 8, 16, 32, 64))

    def forward(self, z):
        if self.ok and torch.is_tensor(z) and z.is_cuda and z.dim() == 4 and z.dtype in (torch.bfloat16, torch.float32) and z.shape[-1] == self.orig[1].weight.shape[1]:
            _count(STATS["sites"], "pair_bias")
            return pair_bias(z, self.orig[0].weight, self.orig[0].bias, self.orig[1].weight, self.eps)
        _count(STATS["sites"], "pair_bias_stock")
        return self.orig(z.float() if torch.is_tensor(z) and z.dtype != torch.float32 else z)


# =====================================================================================================================================
# the sites (provider calls; weights packed once per module instance and cached on it)
# =====================================================================================================================================
class SiteRefused(Exception):
    def __init__(self, site: str, reason: str):
        super().__init__(f"{site}:{reason}")
        self.site, self.reason = site, reason


def _trimul_weights(m):
    w = getattr(m, "_pairfuse_w", None)
    if w is None:
        from opt_core.kernels.fpf_trimul_v4 import generic as G     # the core's unit by its core name (this tree routes / exports nothing for it: the provider's own cell table serves)
        from boltz2_opt.trimul import weights_of                     # the kit's own canonical mapping of the module's parameters (a|b row blocks of p_in/g_in)
        w = G.pack_weights(**weights_of(m), cache_owner=m, cache_key="pairfuse")
        m._pairfuse_w = w
    return w


def _trimul_pick() -> str:
    return (_STATE.get("picks") or DEFAULT_PICKS).get("trimul", "v4")


def trimul_site(m, z4: torch.Tensor, *, outgoing: bool, mask, res_out: torch.Tensor) -> torch.Tensor:
    """TriangleMultiplication{Outgoing,Incoming} site, by the row's provider pick: v4 (below) | core.<word> (the core's provider, by word)."""
    pick = _trimul_pick()
    w = core_word(pick)
    if w is not None:
        return _trimul_site_core(m, z4, outgoing=outgoing, mask=mask, res_out=res_out, word=w)
    return _trimul_site_v4(m, z4, outgoing=outgoing, mask=mask, res_out=res_out)


_CORE_TRIMUL_CACHE: Dict[Any, Any] = {}      # ONE cache for the core's triangle-multiplication face in this process: packed weights keyed by the weight
                                              # tensors' identity (one entry per module), ONE workspace / cast buffer per process (the face keeps one per cache —
                                              # a cache per module would hold a workspace per module)


def _core_trimul_weights(m):
    """The ten canonical TriMul tensors of the module (the kit's own map; the core face packs and caches them per weight set in the process cache)."""
    w = getattr(m, "_pairfuse_core_w", None)
    if w is None:
        from boltz2_opt.trimul import weights_of
        w = m._pairfuse_core_w = dict(weights_of(m))
    return w, _CORE_TRIMUL_CACHE


def _note_core_cell(site: str, N: int, text: str) -> None:
    """Census: which provider row / cell served the site at this token count (once per (site, N); the LEVER line carries it)."""
    key = f"{site}@{int(N)}"
    if key not in STATS["core_cells"]:
        STATS["core_cells"][key] = str(text)
        print(f"[{TAG}] {NAME} CORE-CELL site={site} n={int(N)} {text}", file=sys.stderr, flush=True)


def _trimul_site_core(m, z4: torch.Tensor, *, outgoing: bool, mask, res_out: torch.Tensor, word: str) -> torch.Tensor:
    """Provider trimul=core.<word>: the core's triangle-multiplication face (opt_core.kernels.trimul) BY WORD on the resident z — LN_in, projections,
    gates, contraction, LN_out and the gated output projection inside the selected row, z + update returned in z's dtype and stored into res_out
    (the rows that write in place return res_out's own storage; the others' result is copied in: one pass).  A refusal with the tensors in hand is
    by name (SiteRefused -> the driver fails closed; the probe asked the face's table before the stack started)."""
    from opt_core.kernels import trimul as KT
    w, cache = _core_trimul_weights(m)
    try:
        out = KT.triangle_multiplication(z4, mask, direction="outgoing" if outgoing else "incoming", weights=w, word=word, residual=True, cache=cache)
    except KT.Refusal as e:
        raise SiteRefused("trimul", (f"{e.row}:{e.kind}" if getattr(e, "row", None) else str(e.kind)).replace(" ", "_")[:120])
    if out.data_ptr() != res_out.data_ptr() or out.dtype != res_out.dtype or tuple(out.shape) != tuple(res_out.shape):
        res_out.copy_(out.reshape(res_out.shape))
    _count(STATS["sites"], "trimul_core")
    return res_out


def _trimul_site_v4(m, z4: torch.Tensor, *, outgoing: bool, mask, res_out: torch.Tensor) -> torch.Tensor:
    """Provider trimul=v4: out = z + TriMul(z) written INTO res_out (== z4: in place; K1 reads all of z before K3 writes tile by tile, K3 reads
    each tile's gate/residual before storing it).  z4/res_out [B, N, N, C] contiguous, same dtype (bf16 | fp32).  The launch statement is
    fpf_trimul_v4.generic.trimul_packed's own (cell resolution, warm probe, safety net) with out= supplied."""
    from opt_core.kernels.fpf_trimul_v4 import generic as G, kernels as GK
    w = _trimul_weights(m)
    try:
        cfg = G._check(z4, mask, w, None, None)
        pk = (str(z4.device), int(w["C"]), int(w["D"]), bool(w.get("has_bias")), z4.dtype == torch.float32)
        if pk not in G._PROBES:                                       # the warm numerics probe, once per (device, shape class) — exactly as trimul_packed
            STATS["host_syncs"] += 1                                  # (it synchronises: first call of the class only — see the capture contract)
            if G.probe(z4.device, pk[1], pk[2], pk[3], pk[4]) is not None:
                cfg = G._check(z4, mask, w, None, None)
    except G.TrimulUnsupported as e:
        raise SiteRefused("trimul", e.reason)
    G.COUNTS["generic_calls"] += 1
    out = G.CELLS.run(lambda cc_: GK.trimul_v4_forward(z4, bool(outgoing), mask, w, cc_, eps=1e-5, residual=True, stock_round=G._STOCK_ROUND, pad=16, out=res_out),
                      cfg, z4.device)
    G.COUNTS["generic_served"] += 1
    _count(STATS["sites"], "trimul")
    return out


def _triatt_weights(m, device):
    W = getattr(m, "_opt_pairblock_W", None)                         # shared with boltz2_opt.pairblock (same pack, same attribute): packed once whichever lever meets the module first
    if W is None:
        from opt_core.attn import pair_fused as PF
        from boltz2_opt.pairblock import weights_of
        w = weights_of(m)
        W = PF.pack_triattn_weights(ln_w=w["ln_w"], ln_b=w["ln_b"], w_q=w["w_q"], w_k=w["w_k"], w_v=w["w_v"], w_g=w["w_g"], w_b=w["w_b"],
                                   w_o=w["w_o"], b_o=w["b_o"], n_heads=w["n_heads"], head_dim=w["head_dim"], eps=w["eps"], device=device)
        m._opt_pairblock_W = W
    return W


def _triatt_pick() -> str:
    return (_STATE.get("picks") or DEFAULT_PICKS).get("triatt", DEFAULT_PICKS["triatt"])   # no word applied (a bare driver): the provider's `fast` tier word


def _core_for(n_tokens: int):
    """The attention core of a served block at this token count: the row's pick -- the core provider's face by TIER word (core.<word>: its measured
    cell names the row per card and N; the rows' binding), or an available word by name (k2b, flash, default = the block's own keyed core)."""
    pick = _triatt_pick()
    handed = _STATE["handed_triatt"].get(int(n_tokens))              # a core pick the provider handed to another row / word at this token count (by name, at the probe)
    if handed is not None:
        pick = handed if handed in ("k2b", "flash", "default") else CORE_PREFIX + handed
    if pick == "flash":
        return "flash_triattn", TRIATT_SITE_WORDS["flash"]
    if pick == "default":                                             # the block's own keyed core (opt_core.attn.pair_fused DEFAULT_CORE_TABLE; counted as pf_cores= on the line)
        return PF_DEFAULT_CORE, TRIATT_SITE_WORDS["default"]
    w = core_word(pick)
    if w is not None:
        from boltz2_opt.pairblock import core_callable              # the kit's one binding of the core's triangle-attention face by word (shared with the block adapter)
        return core_callable(w), TRIATT_SITE_WORDS["core"]
    return "k2b", TRIATT_SITE_WORDS["k2b"]


def triatt_site(m, z4: torch.Tensor, *, ending: bool, mask, inplace: bool):
    """TriangleAttention{Starting,Ending}Node site.  inplace=True (bf16 residency): z4 <- z4 + u in the epilogue (ending: u scattered transposed),
    returns z4.  inplace=False (fp32 residency): returns the bf16 update u in z4's frame (a transposed VIEW for the ending node)."""
    from opt_core.attn import pair_fused as PF
    W = _triatt_weights(m, z4.device)
    core, word = _core_for(int(z4.shape[-2]))
    scale = 1.0 / math.sqrt(m.mha.c_hidden)
    try:
        r = PF.tri_attn_block(z4, W, mask=mask, ending=ending, residual=inplace, impl="fpf", core=core, ln="fused", scale=scale)
    except PF.Unsupported as e:
        raise SiteRefused("triatt", e.reason)
    except Exception as e:                                           # the core's provider refused with the tensors in hand (boltz2_opt.pairblock.CoreRefused): by name
        if type(e).__name__ == "CoreRefused":
            raise SiteRefused("triatt", f"core:{getattr(e, 'row', '?')}:{getattr(e, 'kind', '?')}") from None
        raise
    _count(STATS["sites"], word)
    return r


def _transition_weights(m, device):
    T = getattr(m, "_opt_transition_T", None)                        # shared with boltz2_opt.transition
    if T is None:
        from opt_core.attn import pair_fused as PF
        from boltz2_opt.transition import weights_of
        w = weights_of(m)
        T = PF.pack_transition_weights(ln_w=w["ln_w"], ln_b=w["ln_b"], w_a=w["w_a"], w_b=w["w_b"], w_out=w["w_out"], eps=w["eps"], device=device)
        m._opt_transition_T = T
    return T


def _core_transition_weights(m, device):
    W = getattr(m, "_opt_transition_W", None)                        # shared with boltz2_opt.transition (its core.<word> variants pack the same object)
    if W is None:
        from opt_core.kernels import transition as KT
        from boltz2_opt.transition import weights_of
        w = weights_of(m)
        W = KT.pack(w_a=w["w_a"], w_b=w["w_b"], w_o=w["w_out"], ln_w=w["ln_w"], ln_b=w["ln_b"], eps=w["eps"], device=device)
        m._opt_transition_W = W
    return W


def transition_pick() -> str:
    return (_STATE.get("picks") or DEFAULT_PICKS).get("transition", "fpf")


def transition_site(m, z4: torch.Tensor, *, inplace: bool):
    """Transition site (pair): inplace=True -> z4 <- z4 + Transition(z4) in the kernel epilogue (row-local: in place is safe), returns z4;
    inplace=False -> the bf16 update.  Provider = the row's pick: fpf (pair_fused, LN fused) | core.<word> (opt_core.kernels.transition by word)."""
    from opt_core.attn import pair_fused as PF
    pick = transition_pick()
    if pick.startswith("core."):
        from opt_core.kernels import transition as KT
        W = _core_transition_weights(m, z4.device)
        try:
            r, sel = KT.transition(z4, W, word=pick[5:], residual=bool(inplace), out=(z4 if inplace else None), n_tokens=int(z4.shape[-2]), family="pair")
        except KT.Refusal as e:
            raise SiteRefused("transition", "core:%s:%s" % (getattr(e, "row", None) or pick[5:], str(getattr(e, "kind", e))[:60]))
        rw = sel.row + ((":" + sel.variant) if getattr(sel, "variant", None) else "") + (("@" + "".join(f"{k.lower()}{v}" for k, v in sel.cfg.items())) if isinstance(getattr(sel, "cfg", None), dict) else (("@" + str(sel.cfg)) if getattr(sel, "cfg", None) else ""))
        STATS.setdefault("transition_rows", {}); STATS["transition_rows"][rw] = STATS["transition_rows"].get(rw, 0) + 1
        _count(STATS["sites"], "transition")
        return r
    T = _transition_weights(m, z4.device)
    try:
        if inplace:
            r = PF.transition(z4, T, residual=True, ln="fused", out=z4)
        else:
            r = PF.transition(z4, T, residual=False, ln="fused")
    except PF.Unsupported as e:
        raise SiteRefused("transition", e.reason)
    _count(STATS["sites"], "transition")
    return r


# =====================================================================================================================================
# eligibility (once per (module, N, dtype, device): every site's provider is asked BEFORE the first launch; a refusal is the stack's fallback word)
# =====================================================================================================================================
def _autocast_bf16(z) -> bool:
    return bool(z.is_cuda and torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.bfloat16)


def _stack_word(first_layer, z, use_kernels: bool, training: bool) -> Optional[str]:
    """None = served; else the fallback reason (the module's original forward runs, counted)."""
    if training:
        return "training"
    if not use_kernels:
        return "kernels_off"
    if not (torch.is_tensor(z) and z.dim() == 4 and z.dtype in (torch.float32, torch.bfloat16) and _autocast_bf16(z)):
        return "autocast_off"
    C, N = int(z.shape[-1]), int(z.shape[-2])
    if C not in C_SERVED:
        return f"c:{C}"
    if N < NMIN:
        return "below_min_tokens"
    key = (N, str(z.device), _STATE["residency"])
    cache = getattr(first_layer, "_pairfuse_elig", None)
    if cache is None:
        cache = first_layer._pairfuse_elig = {}
    if key in cache:
        return cache[key] or None
    word = _probe_sites(first_layer, z)
    cache[key] = word or ""
    return word


def _probe_core_triatt(mod, zp, word: str):
    """Ask the core's triangle-attention face whether `word` serves this (card, dtype, head_dim, heads, N) — its table, no launch.  Records the
    selected row + cell for the census.  -> (ok, reason)."""
    try:
        from opt_core.kernels import triattn as KA
        cc = torch.cuda.get_device_capability(zp.device)
        N = int(zp.shape[-2]); D = int(mod.mha.c_hidden); H = int(mod.mha.no_heads)
        stack = None
        if tuple(cc) == (9, 0) and (word.split("@")[0] in ("cuda_sm90a", "triattn_native") or word in TIER_PICK_WORDS):
            from opt_core.kernels.triattn import cuda_sm90a as _C
            stack = _C.stack_key()                                   # the CUDA rows' per-ABI prebuilt is refused HERE by name when absent (no_prebuilt:…), not at load
        sel = KA.select(cc, "bf16", D, H, N, word=word, stack=stack)
    except Exception as e:  # noqa: BLE001 — the face's Refusal (by name) or an import problem
        row, kind, fb = getattr(e, "row", None), getattr(e, "kind", None), getattr(e, "fallback", None)
        reason = (f"{row}:{kind}" if kind else f"{type(e).__name__}:{e}").replace(" ", "_")[:160]
        if kind and str(kind).startswith("unknown_word") and word in TIER_PICK_WORDS and word != "fast":   # a tier word this core's face predates (core_word_absent): the pick steps aside BY NAME to the `fast` tier word
            _STATE["handed_triatt"][N] = "fast"
            _note_core_cell("triatt", N, f"handed {word}->fast (core_word_absent:{word})")
            _count(STATS["handed"], f"triatt@{N}:{word}->fast:core_word_absent")
            return _probe_core_triatt(mod, zp, "fast")
        if kind and isinstance(fb, str) and fb in HANDABLE_ROWS:      # the provider names the row to bind instead: the pick steps aside BY NAME to it for this token count
            _STATE["handed_triatt"][N] = fb                          # (never silent: `handed=` on the LEVER line, CORE-CELL line here; the gate holds — a named step-aside, not a fallback)
            _note_core_cell("triatt", N, f"handed {word}->{fb} ({reason})")
            _count(STATS["handed"], f"triatt@{N}:{word}->{fb}:{kind}".replace(" ", "_")[:200])
            if fb in ("k2b", "flash"):
                return True, ""
            return _probe_core_triatt(mod, zp, fb)                    # another face row: ask it the same question
        if kind and str(kind).startswith("no_cell"):                 # the provider's tables carry NO cell for this word at this (card, dtype, head_dim, heads) — not even an inherited
            _STATE["handed_triatt"][N] = "default"                   # one: the site steps aside BY NAME to the block's own keyed core (opt_core.attn.pair_fused core='default': its
            _note_core_cell("triatt", N, f"handed {word}->default ({reason})")   # table's row where listed, the flash_triattn cell elsewhere) for this token count; `handed=` on the LEVER
            _count(STATS["handed"], f"triatt@{N}:{word}->default:{kind}".replace(" ", "_")[:200])   # line names it; the driver keeps serving the stack; the gate holds
            return True, ""
        return False, reason
    _note_core_cell("triatt", N, KA.describe(sel))
    return True, ""


def _probe_core_trimul(mod, zp, word: str):
    """Ask the core's triangle-multiplication face whether `word` serves this (card, precision, c_z, c_hidden, N) — its table, no launch. -> (ok, reason)."""
    try:
        from opt_core.kernels import trimul as KT
        w, _cache = _core_trimul_weights(mod)
        cc = KT._device_cc(zp); prec, _cdt = KT.call_precision(zp)
        dt = prec.split("_")[-1] if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
        res = "fp32" if prec.startswith("f32z_") else None
        N, C, D = int(zp.shape[-2]), int(zp.shape[-1]), int(w["w_ap"].shape[0])
        sels = []
        for d in ("outgoing", "incoming"):
            sel = KT.select(cc, dt, C, D, N, d, word=word, residency=res, stack=KT.stack_word(zp), tf32=(prec == "tf32"))
            KT.admits(sel.row, cc, dt, C, D, N, residency=res, backward=False, residual=True, batch=int(zp.shape[0]),
                      abi=(KT.tx_abi_tag() if str(sel.row).startswith("tx_sm90a") else None))
            sels.append(sel)
    except Exception as e:  # noqa: BLE001
        row, kind = getattr(e, "row", None), getattr(e, "kind", None)
        return False, (f"{row}:{kind}" if kind else f"{type(e).__name__}:{e}").replace(" ", "_")[:160]
    _note_core_cell("trimul", N, " | ".join(KT.describe(s) for s in sels))
    return True, ""


def _probe_sites(layer, z) -> Optional[str]:
    """Ask every provider whether it serves this (N, C, dtype) — no launch."""
    res_dt = torch.bfloat16 if _STATE["residency"] == "bf16" else torch.float32
    N = int(z.shape[-2]); dev = z.device
    zp = torch.empty((1, N, N, int(z.shape[-1])), dtype=res_dt, device=dev)      # a probe tensor of the resident dtype (values never read)
    try:
        try:
            from opt_core.kernels.fpf_trimul_v4 import generic as G
            from opt_core.attn import pair_fused as PF
        except ImportError as e:
            return "import:" + getattr(e, "name", None) or "import"
        if core_word(_trimul_pick()) is not None:
            ok, reason = _probe_core_trimul(layer.tri_mul_out, zp, core_word(_trimul_pick()))
        else:
            ok, reason = G.supported(zp, None, weights=_trimul_weights(layer.tri_mul_out))
        if not ok:
            return _no_cell_word("trimul", reason) or ("trimul:" + str(reason))
        if core_word(_triatt_pick()) is not None:                       # the provider's table first (by word, no launch): the row for this (card, N) or its refusal by name
            ok, reason = _probe_core_triatt(layer.tri_att_start, zp, core_word(_triatt_pick()))
            if not ok:
                return "triatt:" + str(reason)
        core, _ = _core_for(N)
        inplace = _STATE["residency"] == "bf16"
        for mod, ending in ((layer.tri_att_start, False), (layer.tri_att_end, True)):
            ok, reason = PF.supported_triattn(zp, _triatt_weights(mod, dev), None, impl="fpf", core=core, ending=ending, ln="fused", residual=inplace)
            if not ok:
                return "triatt:" + _short(reason)
        if transition_pick().startswith("core."):                      # the provider resolves the word for this (C, hidden, N, card) or names why not; a stock row = not served here, by name
            from opt_core.kernels import transition as KT
            Wc = _core_transition_weights(layer.transition_z, dev)
            try:
                sel = KT.select(transition_pick()[5:], c=int(zp.shape[-1]), hidden=Wc.hidden, n_tokens=N, dtype="bf16", family="pair", residual=inplace, mask=False,
                                device=dev, rows_count=int(zp.numel() // zp.shape[-1]))
            except KT.Refusal as e:
                nc = _no_cell_word("transition", str(getattr(e, "kind", "") or ""))
                if nc:
                    return nc
                return "transition:core:%s:%s" % (getattr(e, "row", None) or transition_pick()[5:], _short(str(getattr(e, "kind", e))))
            if sel.row in getattr(KT, "STOCK_ROWS", ("torch_swiglu", "engine_module", "compile")):   # the face names the stock op for this word on this card (no served cell): the stack takes the
                return f"{NO_CELL}:transition:stock:{sel.row}"                                            # per-layer path BY NAME (declared), never a refusal
        else:
            ok, reason = PF.supported_transition(zp, _transition_weights(layer.transition_z, dev), impl="fpf", ln="fused", residual=inplace)
            if not ok:
                return "transition:" + _short(reason)
    finally:
        del zp
    return None


NO_CELL = "no_cell"                                                   # the stack word's prefix when a site's provider carries no cell for its word on this card: `no_cell:<site>:<detail>`


def _no_cell_word(site: str, reason) -> Optional[str]:
    """A site whose provider's tables carry no cell for its tier word at this (card, shape) — `no_cell…` in the face's refusal kind: the stack
    word `no_cell:<site>:<detail>` (the WHOLE stack steps aside BY NAME to the per-layer path — the module's original forward, where the row's
    per-layer levers serve; a DECLARED reason: counted on the LEVER line as fallback_by=, the gate holds).  None for any other refusal."""
    r = str(reason or "")
    if "no_cell" not in r:
        return None
    detail = r[r.index("no_cell") + len("no_cell"):].lstrip(":").split(" ")[0][:60]
    return f"{NO_CELL}:{site}:{detail}" if detail else f"{NO_CELL}:{site}"


def _short(r: str) -> str:
    if r.startswith("no-cell:"):
        parts = r.split(":")
        return ":".join(parts[:4])
    return r.split(" ")[0][:60]


# =====================================================================================================================================
# the layer drivers
# =====================================================================================================================================
def _draw(device, columnwise: bool, B: int, N: int) -> None:
    """get_dropout_mask's torch.rand_like(z[:, :, 0:1, 0:1]) (or the column-wise z[:, 0:1, :, 0:1]) — the SAME draw (shape, fp32, order) so the
    CUDA Philox offset advances exactly as stock's; the value (mask == 1.0 at eval) is not needed."""
    shape = (B, 1, N, 1) if columnwise else (B, N, 1, 1)
    torch.rand(shape, dtype=torch.float32, device=device)


class _Ctx:
    __slots__ = ("mask_trimul", "mask_att", "inplace", "B", "N")


# ---- the all-ones predicate: once per mask tensor (or given by the caller), never per layer / per stack call
_MASK_PRED: List[Any] = []                                            # [(weakref(tensor), version, value)] — most recently used first, live entries only, at most 8
_MASK_GIVEN: Dict[str, Any] = {"value": None}


def _mask_ver(t) -> int:
    """The tensor's version counter, or -1 for an inference tensor (torch.inference_mode(): no version counter — and no in-place
    mutation by non-inference code either; boltz predicts under inference_mode and builds the mask once per forward)."""
    try:
        return -1 if t.is_inference() else int(t._version)
    except RuntimeError:                                             # 'Inference tensors do not track version counter'
        return -1


def _mask_pred_lookup(pair_mask) -> Optional[bool]:
    ver = _mask_ver(pair_mask)
    for i, (ref, v, val) in enumerate(_MASK_PRED):
        if ref() is pair_mask and v == ver:                          # the SAME live tensor object, unmodified since it was reduced (a live object's identity is unique)
            if i:
                _MASK_PRED.insert(0, _MASK_PRED.pop(i))              # most recently used first
            return val
    return None


def _mask_pred_store(pair_mask, val: bool) -> None:
    _MASK_PRED[:] = [e for e in _MASK_PRED if e[0]() is not None]    # dead tensors leave the cache (host-only bookkeeping)
    _MASK_PRED.insert(0, (weakref.ref(pair_mask), _mask_ver(pair_mask), val))
    del _MASK_PRED[8:]


def mask_is_trivial(pair_mask) -> bool:
    """True when pair_mask is None or all ones.  Given by the caller (`mask_predicate`) -> no device work; else cached per tensor object ->
    no sync; else ONE host sync (the reduction), cached for every later stack call / layer / recycling pass that passes the same tensor."""
    if pair_mask is None:
        return True
    given = _MASK_GIVEN["value"]
    if given is not None:
        STATS["mask_pred_given"] += 1
        return bool(given)
    val = _mask_pred_lookup(pair_mask)
    if val is not None:
        STATS["mask_pred_cached"] += 1
        return val
    val = bool((pair_mask == 1).all())                                # the ONE host sync
    STATS["mask_pred_synced"] += 1; STATS["host_syncs"] += 1
    _mask_pred_store(pair_mask, val)
    return val


@contextlib.contextmanager
def mask_predicate(trivial: Optional[bool]):
    """`with mask_predicate(True|False): model(...)` — the caller states, from the host side, whether every pair mask reaching the served
    stacks inside the block is all ones (True), or must be applied (False); None = decide per tensor as above.  A graph lever sets this per
    item around capture AND around the primed eager call, and keys its captured graph on the value (a graph captured under True must not be
    replayed for a padded item)."""
    prev = _MASK_GIVEN["value"]
    _MASK_GIVEN["value"] = trivial
    try:
        yield
    finally:
        _MASK_GIVEN["value"] = prev


def primed(module, z, pair_mask=None) -> Dict[str, Any]:
    """Whether a served call of `module` (PairformerModule | PairformerNoSeqModule | MSALayer) on z [B,N,N,C] would run WITHOUT any host
    sync or first-call host work — i.e. the shape class has had its eager call: eligibility cached, providers probed, the mask predicate
    given or cached.  {'ok': bool, 'missing': [...]} (host-only; issues nothing)."""
    missing: List[str] = []
    layers = getattr(module, "layers", None)
    first = layers[0] if layers is not None and len(layers) else getattr(module, "pairformer_layer", None)
    if first is None:
        return {"ok": False, "missing": ["not a served module kind"]}
    if torch.is_tensor(z) and z.dim() == 4:
        key = (int(z.shape[-2]), str(z.device), _STATE["residency"])
        if key not in (getattr(first, "_pairfuse_elig", None) or {}):
            missing.append(f"eligibility not cached for N={key[0]} (no eager call of this class yet)")
    else:
        missing.append("z is not a [B,N,N,C] tensor")
    if pair_mask is not None and _MASK_GIVEN["value"] is None and _mask_pred_lookup(pair_mask) is None:
        missing.append("mask predicate neither given (mask_predicate) nor cached for this tensor: the call would issue one host sync")
    if _trimul_pick() == "v4":
        try:
            from opt_core.kernels.fpf_trimul_v4 import generic as G
            want_fp32 = _STATE["residency"] == "fp32"
            if not any(len(pk) >= 5 and pk[1] == 128 and bool(pk[4]) == want_fp32 for pk in G._PROBES):
                missing.append("trimul v4 warm probe has not run for this dtype class")
        except Exception as e:                                        # host-only diagnostic: never raises
            missing.append("trimul probe state unreadable: " + type(e).__name__)
    return {"ok": not missing, "missing": missing}


def _make_ctx(zr: torch.Tensor, pair_mask) -> _Ctx:
    ctx = _Ctx()
    ctx.B, ctx.N = int(zr.shape[0]), int(zr.shape[1])
    ctx.inplace = _STATE["residency"] == "bf16"
    trivial = mask_is_trivial(pair_mask)                              # given | cached per tensor | ONE host sync per new mask tensor (never per layer)
    if trivial:
        STATS["mask_trivial"] += 1
        ctx.mask_trimul = None; ctx.mask_att = None
    else:
        STATS["mask_real"] += 1
        pm = pair_mask if pair_mask.dim() == 3 else pair_mask.unsqueeze(0)
        pm = pm.to(dtype=torch.float32).contiguous()
        ctx.mask_trimul = pm if ctx.B > 1 else pm[0]                # fpf_trimul_v4: [N,N] | [B,N,N] float multiplies a and b (stock: x * mask.unsqueeze(-1))
        ctx.mask_att = pm                                            # pair_fused: [B,N,N] pair mask -> its key-mask view per node
    return ctx


def _pair_sublayers(layer, zr: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
    """The five pair statements of PairformerLayer / PairformerNoSeqLayer on the resident z (stock order, stock RNG draws)."""
    dev = zr.device
    B, N = ctx.B, ctx.N
    # z = z + dropout * self.tri_mul_out(z, mask=pair_mask)
    _draw(dev, False, B, N)
    trimul_site(layer.tri_mul_out, zr, outgoing=True, mask=ctx.mask_trimul, res_out=zr)
    # z = z + dropout * self.tri_mul_in(z, mask=pair_mask)
    _draw(dev, False, B, N)
    trimul_site(layer.tri_mul_in, zr, outgoing=False, mask=ctx.mask_trimul, res_out=zr)
    # z = z + dropout * self.tri_att_start(z, mask=pair_mask, ...)
    _draw(dev, False, B, N)
    if ctx.inplace:
        triatt_site(layer.tri_att_start, zr, ending=False, mask=ctx.mask_att, inplace=True)
    else:
        zr.add_(triatt_site(layer.tri_att_start, zr, ending=False, mask=ctx.mask_att, inplace=False))
    # z = z + dropout * self.tri_att_end(z, mask=pair_mask, ...)   (column-wise draw)
    _draw(dev, True, B, N)
    if ctx.inplace:
        triatt_site(layer.tri_att_end, zr, ending=True, mask=ctx.mask_att, inplace=True)
    else:
        zr.add_(triatt_site(layer.tri_att_end, zr, ending=True, mask=ctx.mask_att, inplace=False))
    # z = z + self.transition_z(z)
    if ctx.inplace:
        transition_site(layer.transition_z, zr, inplace=True)
    else:
        zr.add_(transition_site(layer.transition_z, zr, inplace=False))
    return zr


def _seq_sublayers(layer, s: torch.Tensor, zr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """PairformerLayer's sequence statements, verbatim except `z=z.float()` -> `z=zr` (proj_z is the fused instance module: it reads the
    resident z in one pass and returns the fp32 bias the stock statement produced)."""
    att = layer.attention
    if not isinstance(att.proj_z, FusedProjZ):
        if isinstance(att.proj_z, torch.nn.Sequential) and len(att.proj_z) == 3:
            att.proj_z = FusedProjZ(att.proj_z)
            _STATE["proj_z_patched"].append(att)
    zin = zr if isinstance(att.proj_z, FusedProjZ) else zr.float()
    with torch.autocast("cuda", enabled=False):
        s_normed = layer.pre_norm_s(s.float())
        s = s.float() + layer.attention(s=s_normed, z=zin, mask=mask.float(), k_in=s_normed)
        s = s + layer.transition_s(s)
        s = layer.s_post_norm(s)
    return s


def _enter(z: torch.Tensor) -> torch.Tensor:
    """The driver-owned resident buffer for this stack call (never the caller's tensor)."""
    res_dt = torch.bfloat16 if _STATE["residency"] == "bf16" else torch.float32
    return z.to(dtype=res_dt, memory_format=torch.contiguous_format, copy=True)


def _exit(zr: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return zr if zr.dtype == like.dtype else zr.to(like.dtype)


def _alloc_snapshot(tag: str) -> None:
    try:
        st = torch.cuda.memory_stats()
        STATS["alloc"][tag] = {"segments": st.get("segment.all.current"), "num_alloc_retries": st.get("num_alloc_retries"),
                               "reserved_gib": round(st.get("reserved_bytes.all.current", 0) / 2**30, 3)}
    except Exception:
        pass


def _served_stack(module_name: str, layers, s, z, mask, pair_mask):
    """Drive every layer of one pair stack on the resident z. Returns (s, z_out)."""
    STATS["stack_served"] += 1; _count(STATS["served_by"], module_name)
    zr = _enter(z)
    ctx = _make_ctx(zr, pair_mask)
    _count(STATS["shapes"], f"{module_name}:{ctx.N}x{int(zr.shape[-1])}")
    first = STATS["first_served"] is None
    if first:
        _alloc_snapshot("before_first_stack")
    for i, layer in enumerate(layers):
        try:
            zr = _pair_sublayers(layer, zr, ctx)
        except SiteRefused as e:                                     # a provider refused AFTER the probe said yes (cannot happen on a healthy box): fail closed, by name
            _count(STATS["errors"], f"refused_mid_stack:{e.site}:{e.reason}")
            raise RuntimeError(f"[{TAG}] {NAME}: provider refused mid-stack ({e.site}: {e.reason}) at layer {i} of {module_name} — the resident chain cannot fall back mid-layer") from None
        except Exception as e:                                       # a kernel error: counted, then raised (no silent stock rescue of a half-updated resident z)
            _count(STATS["errors"], type(e).__name__)
            raise
        if s is not None:
            s = _seq_sublayers(layer, s, zr, mask)
            STATS["layers"] += 1
        else:
            STATS["noseq_layers"] += 1
        if first and i == 0:
            _alloc_snapshot("after_first_layer")
    if first:
        _alloc_snapshot("after_first_stack")
        STATS["first_served"] = {"stack": module_name, "N": ctx.N, "B": ctx.B, "layers": len(layers), "residency": _STATE["residency"],
                                 "mask": "trivial" if ctx.mask_att is None else "real", "core": _core_for(ctx.N)[1], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(f"[{TAG}] {NAME} FIRST STACK served: {STATS['first_served']}", file=sys.stderr, flush=True)
    return s, _exit(zr, z)


def _fallback(reason: str) -> None:
    _count(STATS["fallback"], reason)
    if STATS["fallback"][reason] == 1:
        print(f"[{TAG}] {NAME} fallback:{reason} — this pair stack takes the module's original forward (counted)", file=sys.stderr, flush=True)


# ---- the three class-level replacements ---------------------------------------------------------------------------------------------
def pfm_forward(self, s, z, mask, pair_mask, use_kernels: bool = False):
    """PairformerModule.forward (trunk 64 blocks, confidence 8 blocks)."""
    STATS["stack_calls"] += 1
    word = _stack_word(self.layers[0], z, use_kernels, self.training) if len(self.layers) else "empty"
    if word is not None:
        _fallback(word)
        return _STATE["orig"]["pfm_fwd"](self, s, z, mask, pair_mask, use_kernels)
    return _served_stack("PairformerModule", self.layers, s, z, mask, pair_mask)


def pfnm_forward(self, z, pair_mask, use_kernels: bool = False):
    """PairformerNoSeqModule.forward (template stacks: C=64 -> handed back by name)."""
    STATS["stack_calls"] += 1
    word = _stack_word(self.layers[0], z, use_kernels, self.training) if len(self.layers) else "empty"
    if word is not None:
        _fallback(word)
        return _STATE["orig"]["pfnm_fwd"](self, z, pair_mask, use_kernels)
    _, z_out = _served_stack("PairformerNoSeqModule", self.layers, None, z, None, pair_mask)
    return z_out


def msal_forward(self, z, m, token_mask, msa_mask, chunk_heads_pwa: bool = False, chunk_size_transition_z: int = None,
                 chunk_size_transition_msa: int = None, chunk_size_outer_product: int = None, chunk_size_tri_attn: int = None,
                 use_kernels: bool = False):
    """MSALayer.forward (trunkv2.py:714-758): the four stock statements verbatim; the pair layer (the last) driven on the resident z."""
    from boltz.model.layers.dropout import get_dropout_mask
    # Communication to MSA stack
    msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)
    m = m + msa_dropout * self.pair_weighted_averaging(m, z, token_mask, chunk_heads_pwa)
    m = m + self.msa_transition(m, chunk_size_transition_msa)
    z = z + self.outer_product_mean(m, msa_mask, chunk_size_outer_product)
    # Compute pairwise stack
    STATS["stack_calls"] += 1
    layer = self.pairformer_layer
    word = _stack_word(layer, z, use_kernels, self.training)
    if word is not None:
        _fallback(word)
        z = layer(z, token_mask, chunk_size_tri_attn, use_kernels=use_kernels)
    else:
        _, z = _served_stack("MSALayer", (layer,), None, z, None, token_mask)
    return z, m


# =====================================================================================================================================
# apply / remove / report
# =====================================================================================================================================
def apply(word: str) -> List[str]:
    if _STATE["applied"]:
        return [NAME]
    r, picks = parse_word(word)
    for site, retired in RETIRED_PICKS.items():                       # a pick whose kit copy left this tree: refused BY NAME before any hook is installed (worker_launch: REFUSED, exit 3) — never a silent re-binding
        if picks.get(site) in retired:
            raise RuntimeError(f"{NAME}: {site}={picks[site]} refused by name — {site}:{picks[site]}:kit_copy_retired (the bz2_trimul kit copy left this tree at boltz2 0.3.12; "
                               f"the core provider's rows serve its cells: {site}=core.fast / core.v4 / core.native)")
    if r == "fp32" and not FP32_ROWS_ADMITTED:                       # source constant only: the kit consults no core gate variable
        raise RuntimeError(f"{NAME}: residency fp32 refused by name — its LayerNorm-from-fp32-z prologue/transition rows of opt_core.attn.pair_fused "
                           f"(v3_fp32z / v1_fp32x) are candidate rows, not admitted on this core (CORE-REQ pending); the kit sets no core gate variable")
    from boltz.model.layers import pairformer as PFM
    from boltz.model.modules import trunkv2 as TR
    _STATE["orig"] = {"pfm_fwd": PFM.PairformerModule.forward, "pfnm_fwd": PFM.PairformerNoSeqModule.forward, "msal_fwd": TR.MSALayer.forward}
    PFM.PairformerModule.forward = pfm_forward
    PFM.PairformerNoSeqModule.forward = pfnm_forward
    TR.MSALayer.forward = msal_forward
    _STATE["residency"] = r; _STATE["picks"] = picks; _STATE["word"] = str(word)
    _STATE["applied"] = True; _STATE["t_apply"] = time.time()
    print(f"[{TAG}] {NAME} APPLIED residency={r} providers={providers_word()} "
          f"hooks=PairformerModule.forward,PairformerNoSeqModule.forward,MSALayer.forward proj_z=instance(FusedProjZ, on first served layer)", file=sys.stderr, flush=True)
    return [NAME]


def pf_cores_word() -> str:
    """The attention cores the core block served through its own dispatch in this process (`default=flash_triattn:<n>`,
    `default=tier:fast=<row>:<n>`, ...; opt_core.attn.pair_fused SERVED_CORES) — the census of the `default` pick; `-` when none."""
    try:
        from boltz2_opt.pairblock import pf_cores_word as _w
        return _w()
    except Exception:  # noqa: BLE001
        return "-"


def providers_word() -> str:
    """The providers credited on the LEVER line — `trimul:<impl>,triatt:<impl>,transition:<impl>,pairbias:<impl>` (+ `[<row>=<n>;...]` after a
    transition=core.<word> pick: the provider rows / launches that served, opt_core.kernels.transition Selection)."""
    p = _STATE.get("picks") or DEFAULT_PICKS
    out = ",".join(f"{k}:{p[k]}" for k in ("trimul", "triatt", "transition", "pairbias"))
    tr = STATS.get("transition_rows") or {}
    if tr:
        out += "[" + ";".join(f"{k}={n}" for k, n in sorted(tr.items())) + "]"
    return out


def remove() -> None:
    _MASK_PRED.clear(); _MASK_GIVEN["value"] = None
    if not _STATE["applied"]:
        return
    from boltz.model.layers import pairformer as PFM
    from boltz.model.modules import trunkv2 as TR
    O = _STATE["orig"]
    PFM.PairformerModule.forward = O["pfm_fwd"]; PFM.PairformerNoSeqModule.forward = O["pfnm_fwd"]; TR.MSALayer.forward = O["msal_fwd"]
    for att in _STATE["proj_z_patched"]:
        if isinstance(att.proj_z, FusedProjZ):
            att.proj_z = att.proj_z.orig
    _STATE["proj_z_patched"].clear()
    _STATE["applied"] = False


def composed_ok() -> Optional[str]:
    """None if the hooks are in force — directly, through wrappers that set __wrapped__ (followed transitively), or through boltz_trunk_levers'
    mask2 scope (its recorded orig, again followed through __wrapped__); else what displaced them."""
    from boltz.model.layers import pairformer as PFM
    from boltz.model.modules import trunkv2 as TR
    cur = {"pfm_fwd": PFM.PairformerModule.forward, "pfnm_fwd": PFM.PairformerNoSeqModule.forward, "msal_fwd": TR.MSALayer.forward}
    mine = {"pfm_fwd": pfm_forward, "pfnm_fwd": pfnm_forward, "msal_fwd": msal_forward}
    lev = sys.modules.get("boltz_trunk_levers")
    lev_orig = (getattr(lev, "_STATE", {}) or {}).get("orig", {}) if lev is not None else {}
    def reaches(f, target, hops: int = 16) -> bool:                  # f itself, or what it wraps (functools' __wrapped__ convention), transitively
        while f is not None and hops:
            if f is target:
                return True
            f = getattr(f, "__wrapped__", None); hops -= 1
        return False
    bad = []
    for k in mine:
        # in force directly; or under wrappers that declare what they wrap (__wrapped__, e.g. a graph lever's capture wrapper); or under mask2's
        # _scoped wrapper (boltz_trunk_levers records the callable it wrapped as its orig) — possibly itself around such wrappers
        if reaches(cur[k], mine[k]) or reaches(lev_orig.get(k), mine[k]):
            continue
        bad.append(f"{k}={getattr(cur[k], '__module__', '?')}.{getattr(cur[k], '__qualname__', '?')}")
    return None if not bad else ";".join(bad)


DECLARED = ("kernels_off", "below_min_tokens", "training")          # + every c:<C> word (the template stack) + every no_cell:<site>:<detail> word (a site's provider carries no cell on this card: the stack takes the per-layer path by name): the reasons a healthy run may show


def declared(reason: str) -> bool:
    """A fallback reason a healthy run may show: the DECLARED words, the template stack's c:<C>, a site's no_cell:<site>:<detail> hand-off."""
    r = str(reason)
    return r in DECLARED or r.startswith("c:") or r.startswith(NO_CELL + ":")


def gate() -> Dict[str, Any]:
    """Fail-closed: refused when not applied, when displaced, on any error, or on a fallback for an UNDECLARED reason (a core's refusal word).
    Declared idle reasons only (every stack of the run below NMIN / kernels off / the template dim / a site with no provider cell on this card,
    handed to the per-layer path by name) = installed and idle, ok.  The rule is
    over BODY executions (every executed stack body was served or took a declared stock path); it never compares counts with the number of
    calls the model made, so a body captured once and replayed by a graph lever keeps the gate's meaning (replays are that lever's census)."""
    if not _STATE["applied"]:
        return {"ok": False, "idle": False, "reason": "not applied"}
    disp = composed_ok()
    if disp:
        return {"ok": False, "idle": False, "reason": "displaced:" + disp}
    if STATS["errors"]:
        return {"ok": False, "idle": False, "reason": "errors:" + ",".join(sorted(STATS["errors"]))}
    undeclared = [k for k in STATS["fallback"] if not declared(k)]
    if undeclared:
        return {"ok": False, "idle": False, "reason": "fallback:" + ",".join(sorted(undeclared))}
    if STATS["stack_calls"] > 0 and STATS["stack_served"] == 0:
        return {"ok": True, "idle": True, "reason": "idle: every pair stack of the run took a declared stock path"}
    return {"ok": True, "idle": False, "reason": None}


def line() -> str:
    g = gate()
    state = "on" if (g["ok"] and not g["idle"] and STATS["stack_served"] > 0) else ("skipped" if g["ok"] else "refused")
    fb = ",".join(f"{k}:{v}" for k, v in sorted(STATS["fallback"].items())) or "none"
    sites = ",".join(f"{k}:{v}" for k, v in sorted(STATS["sites"].items()) if v) or "none"
    shapes = ",".join(f"{k}:{v}" for k, v in sorted(STATS["shapes"].items())) or "none"
    return (f"[{TAG}] LEVER name={NAME} state={state} residency={_STATE['residency']} providers={providers_word()} pf_cores={pf_cores_word()} served={STATS['stack_served']} calls={STATS['stack_calls']} "
            f"layers={STATS['layers']} noseq_layers={STATS['noseq_layers']} fallback_by={fb} sites={sites} mask=trivial:{STATS['mask_trivial']},real:{STATS['mask_real']} "
            f"host_syncs={STATS['host_syncs']} mask_pred=synced:{STATS['mask_pred_synced']},cached:{STATS['mask_pred_cached']},given:{STATS['mask_pred_given']} "
            f"shapes={shapes} core_cells={_core_cells_word()} handed={_tokjoin(STATS['handed'])} core_routes={_tokjoin(_core_routes())} "
            f"errors={','.join(sorted(STATS['errors'])) or 'none'} gate={'ok' if g['ok'] else 'REFUSED:' + str(g['reason'])}")


def _tokjoin(d) -> str:
    """{k: v} -> 'k=v;…' with spaces as underscores ('-' when empty): one LEVER-line token."""
    return (";".join(f"{k}={v}" for k, v in sorted((d or {}).items())) or "-").replace(" ", "_")


def _core_routes() -> Dict[str, str]:
    """The CUDA rows' served route per (token count, operand layout) as the core's package reported it at the first such call (boltz2_opt.pairblock)."""
    try:
        from boltz2_opt.pairblock import core_routes
        return dict(core_routes())
    except Exception:  # noqa: BLE001
        return {}


def _core_cells_word() -> str:
    """`-` when no site runs through the core's faces; else `<site>@<N>=<row>[<cell>];…` (the provider's own census words, spaces as _)."""
    cc = STATS.get("core_cells") or {}
    return ";".join(f"{k}={str(v).replace(' ', '_')}" for k, v in sorted(cc.items())) or "-"


def report() -> Dict[str, Any]:
    return {"applied": [NAME] if _STATE["applied"] else [], "disabled": {}, "line": line() if _STATE["applied"] else None, "gate": gate(),
            "residency": _STATE["residency"], "stats": {k: (dict(v) if isinstance(v, dict) else v) for k, v in STATS.items()},
            "patched": ["boltz.model.layers.pairformer.PairformerModule.forward", "boltz.model.layers.pairformer.PairformerNoSeqModule.forward",
                        "boltz.model.modules.trunkv2.MSALayer.forward", f"AttentionPairBias.proj_z x{len(_STATE['proj_z_patched'])} (instance)"] if _STATE["applied"] else [],
            "providers": providers_word(), "pf_cores": pf_cores_word(), "picks": dict(_STATE.get("picks") or {}), "word": _STATE.get("word"), "fp32_rows_admitted": FP32_ROWS_ADMITTED,
            "handed": dict(STATS.get("handed") or {}), "core_routes": _core_routes(),
            "counting": "body executions (a captured body replayed by a graph lever is counted once here; replays are that lever's census)",
            "torch": torch.__version__}
