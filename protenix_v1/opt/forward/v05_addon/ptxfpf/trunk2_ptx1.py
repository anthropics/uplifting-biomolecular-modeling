"""trunk2_ptx1.py — the protenix 1.1.0 adapter of the fused TRUNK-side kernels: lever words `pfattn`, `opm_fused`, `pwa_fused` of the kit arm
(levers_ptx1.apply: `<trimul>[+lever...]`). TOLERANCE class, all three (fast + big tiers).

Kernels. `pfattn` binds the shared core's attention-with-pair-bias PROVIDER (opt_core.kernels.apb) by TIER word — the kit mode's own word
(fast | big | exact: tier_word()) — and owns no kernel and no cell: for each call class the provider's table (APB_CELLS) names the module
boundary's row (`composed` on the served cards: its bias-producer row + its attention-core row, torch around them; `torch_module` = the library
statement itself, taken BY NAME) and serves the producer (pair_bias_planes) and the core (pair_bias_attention); a row's Refusal is a named
step-aside to the module's own statement for that call. `opm_fused` / `pwa_fused` run the shared core's MSA-module Triton kernels
(opt_core.ops.msa_fused.msa_triton: ln_linear / opm_out / pwa_ln_vg / pwa_out2; launch cells CELLS below; lib/protenix_fpf_msa is the package
face); pwa_fused's z-path (Linear(LayerNorm(z)) head-major) is the same provider's producer row by tier word. Sites (the pinned wheel's,
stock/protenix-1.1.0-py3-none-any.whl):

  pfattn     transformer.py AttentionPairBias.standard_multihead_attention(q, kv, z, inplace_safe=False, enable_efficient_fusion=False) on the
             has_s=False modules OUTSIDE the DiffusionModule = the trunk PairformerStack's 48 blocks (PairformerBlock: AttentionPairBias(has_s=False,
             create_offset_ln_z=..., n_heads 16, c_a = c_s 384 -> c_hidden 24, gating, q bias)) + the ConfidenceHead's 4-block pairformer
             (skip_amp.confidence_head False: bf16 autocast like the trunk). Stock: bias = linear_nobias_z(layernorm_z(z)) (LayerNorm kernel +
             Linear + permute copy), then primitives.Attention (q/k/bias .float() copies, F.scaled_dot_product_attention under autocast,
             transposes, sigmoid-gate chain, linear_o). Lever: the provider's producer row over z (LayerNorm + the 16-head projection in one pass,
             head-major planes) and its attention-core row (bias shared, sigmoid gate) on stock's own bf16 q/k/v/g projections, then stock's
             linear_o. TOLERANCE (the rows' reduction order).
  opm_fused  triangular/layers.py OuterProductMean.forward(m, mask=None, chunk_size=None, inplace_safe=False) on the MSA module's 4 blocks
             (MSABlock.outer_product_mean_msa; c_m 64, c_hidden 32, c_z 128; the eval call MSABlock makes: mask None, chunk_size = the runner's).
             Lever: fused LayerNorm(m)+[linear_1|linear_2] (ln_linear, one pass over m) -> stock's OWN einsum statement per row chunk (the same
             cuBLAS GEMM) -> opm_out: linear_out on the einsum's native [b,c,d,e] layout (no permute copy) with + bias and / norm in the epilogue
             (norm = stock's einsum(mask, mask) + eps statement).
  pwa_fused  pairformer.py MSAPairWeightedAveraging.forward(m, z) on the 3 MSAStack modules (8 heads x c 32, c_m 64, c_z 128), called per MSA row
             chunk by MSAStack.inference_forward. Lever: fused LayerNorm(m)+[linear_mv|linear_mg] writing v head-major (pwa_ln_vg) so the
             pair-weighted average is one torch.matmul (cuBLAS bmm, fp32 accumulation; stock's einsum needs two permute copies), z-path = the
             provider's producer row (LayerNorm(z)+linear_z, fp32 logits) + softmax over j, then pwa_out2: sigmoid(g) * wv -> linear_out in one
             kernel.

Envelope (every miss steps aside BY NAME for that call — counted `stock:<reason>`, the module's own statement answers it; never silent, never a
refusal): CUDA tensors; bf16 autocast on (the kit's `--dtype bf16` runs; activations stored fp32 or bf16 — an fp32 / no-autocast run keeps the
stock fp32 statements: `stock:no_autocast`; autocast on at another compute dtype (`--dtype fp32 | fp16`): `stock:autocast_dtype=<dtype>`, read
before any launch; an fp16 tensor `stock:dtype=float16`); eval mode; pfattn: self-attention (kv is q), no efficient-fusion flag, z square
[N, N, c_z] with at most size-1 leading dims (a batched pair tensor -> `stock:z_batch`), q [.., N, c_a]; opm: mask None, m [S, N, c_m]; pwa:
m [S, N, c_m], z [N, N, c_z]. A kernel error is counted `error:<Type>`, printed (first two), and the stock statement answers the call (an
out-of-memory is re-raised: opt_core.oom). Under the row-sharded line (`--n_gpu` P > 1, protenix_v1_opt.tp) the sharded statements call the
modules' projections directly and never enter these three methods: installed, idle, recorded replaced_by_rowpair by the big line
(big.ROWPAIR_REPLACES).

Cards: cc >= MIN_CC (8.0: bf16 MMA) engages; below it the levers install on nothing, named `arch=`. pfattn's rows and cells are the provider's per
card; the MSA kernels' launch cells are CELLS below (rows for cc 9.0 at this model's geometry; a card without a row engages the 9.0 row, named
`cell=9.0 on sm_XY`); their shared-memory needs are under sm_80's 163 KB.

API (levers_ptx1.set_trunk2 calls apply): apply(model, pfattn=, opm_fused=, pwa_fused=) -> describe(); install_pfattn / install_opm / install_pwa;
uninstall(); lever_facts(word); describe(). Importable without torch: the routing decisions are pure functions (pf_route / opm_route / pwa_route /
card_cell), the install path imports torch / triton / protenix / the packages.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict

__version__ = "1.1.0"
APB_PROVIDER = "opt_core.kernels.apb"                    # the shared core's attention-with-pair-bias PROVIDER: pfattn = its producer + core rows by TIER word (fast | big | exact = the mode); pwa_fused's z-path = its producer rows
APB_PACKAGE = "opt_core.kernels.apb.fpf_apb"             # one of the provider's carried row packages (pf_triton / apb_triton; the sampler words' package, apb_ptx1.PACKAGE) — named here for the account only
MSA_PACKAGE = "protenix_fpf_msa"                        # lib/protenix_fpf_msa: the MSA-module lever package (cells, load check; README.md beside it); its kernels are the shared core's opt_core.ops.msa_fused.msa_triton (ln_linear / opm_out / pwa_ln_vg / pwa_out2)
PACKAGES = (APB_PROVIDER, MSA_PACKAGE)
TIER_WORDS = ("fast", "big", "exact")                   # the provider tier word = the kit mode (PROTENIX_V1_OPT); anything else reads `fast`
LEVERS = ("pfattn", "opm_fused", "pwa_fused")           # the arm words this adapter answers (levers_ptx1.LEVER_NAMES carries them)
KIND = {"pfattn": "pf", "opm_fused": "opm", "pwa_fused": "pwa"}

# Geometry read from the pinned wheel (configs/configs_base.py: c_s 384, c_z 128, pairformer n_heads 16; msa_module c_m 64; OuterProductMean c_hidden 32;
# MSAPairWeightedAveraging n_heads 8, c 32) — the geometry the cells cover. Another geometry is not installed on (named), never served with a guess.
PF_HEADS, PF_HEAD_DIMS = 16, (16, 24, 32, 48, 64)       # the pairformer geometry the provider's pf cells list (16 heads; head dims its core rows tile)
MIN_CC = (8, 0)                                         # Ampere and newer engage (bf16 MMA)
SERVED_DTYPES = ("bfloat16", "float32")                # served under bf16 AUTOCAST (activations stored fp32 or bf16): the kernels round LayerNorm outputs / operands to bf16 exactly where stock's autocast Linears do; an fp16 / fp64 tensor or a no-autocast (fp32) run keeps the stock statement, by name

# The launch cells of the MSA kernels per compute capability, for THIS model's geometry (c_m 64, c_z 128, 8 heads x 32); a card >= MIN_CC
# without a row engages the REFERENCE_CC row, named.
CELLS = {                                                 # launch cells of the carried MSA kernels per cc
    "opm": {"9.0": {"ln_rows": 64, "td": 64, "cg": 1, "num_warps": 4, "num_stages": 3}},       # cc 9.0: ln_linear rows per program; opm_out d-tile, c-slices per MMA step, warps, stages
    "pwa": {"9.0": {"mb": 8, "nb": 8, "ln_warps": 4, "out_mb": 8, "out_nb": 8, "out_warps": 4}},   # cc 9.0: pwa_ln_vg tile (msa rows x tokens) and warps; pwa_out2 tile and warps
}                                                         # pfattn has NO kit cell: the provider's table (opt_core.kernels.apb APB_CELLS) selects producer and core rows per (card, dtype, N) by tier word
REFERENCE_CC = "9.0"

SERVED_PREFIX, ASIDE_PREFIX, ERROR_PREFIX = "t2:", "stock:", "error:"
COUNTS = {"pf": defaultdict(int), "opm": defaultdict(int), "pwa": defaultdict(int)}      # per-call census words (levers_ptx1.COUNTS["pf"|"opm"|"pwa"] are these dicts)
STATE = {k: {"requested": False, "installed_on": 0, "found": 0, "cell": None, "cell_key": None, "named": None, "calls": 0, "chunks": 0, "tier": None, "rows": {}}
         for k in ("pf", "opm", "pwa")}
_INSTALLED = {"pf": [], "opm": [], "pwa": []}           # (module, attr, had_instance_attr, previous) for uninstall
_PRINTED = defaultdict(int)


# ---------------------------------------------------------------- pure routing (no torch needed)
def card_cell(kind: str, cc):
    """(cell dict | None, cell_key | None, named | None) for lever kind on a card of capability `cc` ((major, minor), or None = no CUDA).
    None cell = the lever installs on nothing (named says why); a card >= MIN_CC without a row engages the REFERENCE_CC row and is named."""
    if cc is None:
        return None, None, "no CUDA device"
    cc = (int(cc[0]), int(cc[1]))
    if cc < MIN_CC:
        return None, None, "arch=sm_%d%d below sm_%d%d (no bf16 tensor-core MMA for the kernels' dots)" % (cc + MIN_CC)
    key = "%d.%d" % cc
    if kind == "pf":                                     # no kit cell: the provider selects per call class (its census names the rows); the key names the card only
        return {}, key, None
    table = CELLS[kind]
    named = None
    if key not in table:
        named = "cell=%s on sm_%d%d (unmeasured card: engaged with the cc %s cell)" % ((REFERENCE_CC,) + cc + (REFERENCE_CC,))
        key = REFERENCE_CC
    return dict(table[key]), key, named


def _pow2(n) -> bool:
    n = int(n)
    return n >= 16 and n & (n - 1) == 0


def pf_geometry_word(heads: int, head_dim: int, gating: bool, q_bias: bool, c_z: int):
    """None when the pairformer attention geometry is one the cells serve, else the `geometry=` word naming it."""
    if heads == PF_HEADS and head_dim in PF_HEAD_DIMS and gating and q_bias and _pow2(c_z):
        return None
    return "geometry=%dx%d:c_z%d%s%s" % (heads, head_dim, c_z, "" if gating else ":nogate", "" if q_bias else ":noqbias")


def opm_geometry_word(c_m: int, c_hidden: int, c_z: int, out_in: int):
    if _pow2(c_m) and _pow2(c_hidden) and _pow2(c_z) and out_in == c_hidden * c_hidden:
        return None
    return "geometry=c_m%d:c%d:c_z%d" % (c_m, c_hidden, c_z)


def pwa_geometry_word(c_m: int, heads: int, c: int, c_z: int):
    if _pow2(c_m) and _pow2(heads * c) and _pow2(c_z) and heads > 0:
        return None
    return "geometry=c_m%d:%dx%d:c_z%d" % (c_m, heads, c, c_z)


def _common_route(dtype: str, is_cuda: bool, autocast: bool, training: bool, autocast_dtype: str = "bfloat16"):
    if not is_cuda:
        return "stock:device"
    if training:
        return "stock:training"
    if dtype not in SERVED_DTYPES:
        return "stock:dtype=%s" % dtype
    if not autocast:
        return "stock:no_autocast"
    if autocast_dtype != "bfloat16":                     # --dtype fp32 | fp16: autocast on at another compute dtype — the kernels are bf16: the stock statement BY NAME,
        return "stock:autocast_dtype=%s" % autocast_dtype   # decided here before any launch (never a caught kernel assertion per call)
    return None


def pf_route(q_shape, kv_is_q: bool, z_shape, efficient_fusion: bool, *, dtype: str = "bfloat16", z_dtype: str = "bfloat16",
             is_cuda: bool = True, autocast: bool = True, training: bool = False, autocast_dtype: str = "bfloat16"):
    """None when pfattn serves this standard_multihead_attention call, else the `stock:<reason>` word. q [.., N, c_a]; z [.., N, N, c_z] whose
    leading dims are all 1 (one pair tensor shared by q's leading rows) — a batched pair tensor is the stock statement's (`stock:z_batch`)."""
    w = _common_route(dtype, is_cuda, autocast, training, autocast_dtype)
    if w:
        return w
    if z_dtype not in SERVED_DTYPES:
        return "stock:z_dtype=%s" % z_dtype
    if efficient_fusion:
        return "stock:efficient_fusion"
    if not kv_is_q:
        return "stock:cross"
    q_shape = tuple(int(x) for x in q_shape); z_shape = tuple(int(x) for x in z_shape)
    if len(q_shape) < 2 or len(z_shape) < 3:
        return "stock:q_shape" if len(q_shape) < 2 else "stock:z_shape"
    N = q_shape[-2]
    if z_shape[-3] != z_shape[-2]:
        return "stock:z_shape"
    if any(x != 1 for x in z_shape[:-3]):
        return "stock:z_batch"
    if z_shape[-2] != N:
        return "stock:z_len"
    if N < 1:
        return "stock:q_shape"
    return None


def opm_route(m_shape, mask_given: bool, *, dtype: str = "bfloat16", is_cuda: bool = True, autocast: bool = True, training: bool = False,
              fp16: bool = False, autocast_dtype: str = "bfloat16"):
    w = _common_route(dtype, is_cuda, autocast, training, autocast_dtype)
    if w:
        return w
    if fp16:
        return "stock:fp16"
    if mask_given:
        return "stock:mask"
    if len(tuple(m_shape)) != 3:
        return "stock:m_dim=%d" % len(tuple(m_shape))
    return None


def pwa_route(m_shape, z_shape, *, dtype: str = "bfloat16", z_dtype: str = "bfloat16", is_cuda: bool = True, autocast: bool = True,
              training: bool = False, autocast_dtype: str = "bfloat16"):
    w = _common_route(dtype, is_cuda, autocast, training, autocast_dtype)
    if w:
        return w
    if z_dtype not in SERVED_DTYPES:
        return "stock:z_dtype=%s" % z_dtype
    m_shape = tuple(int(x) for x in m_shape); z_shape = tuple(int(x) for x in z_shape)
    if len(m_shape) != 3:
        return "stock:m_dim=%d" % len(m_shape)
    if len(z_shape) != 3 or z_shape[0] != z_shape[1]:
        return "stock:z_shape"
    if z_shape[0] != m_shape[1]:
        return "stock:z_len"
    return None


def served_word(kind: str, cell_key: str) -> str:
    return "%s%s@%s" % (SERVED_PREFIX, kind, cell_key)


# ---------------------------------------------------------------- install (torch)
def tier_word(environ=None) -> str:
    """The provider tier word for this process: the kit mode (stack.ENV_MODE = PROTENIX_V1_OPT: fast | big | exact); a process without the
    kit's mode variable (the adapter imported outside the kit) reads `fast`."""
    import os
    m = str((environ if environ is not None else os.environ).get("PROTENIX_V1_OPT", "") or "").strip()
    return m if m in TIER_WORDS else "fast"


def _cc():
    import torch
    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability())


def _dt(t) -> str:
    return str(t.dtype).rsplit(".", 1)[-1]


def _autocast_on() -> bool:
    import torch
    return bool(torch.is_autocast_enabled())


def _autocast_dtype() -> str:
    """The CUDA autocast compute dtype's word ('bfloat16' | 'float32' | 'float16'); 'bfloat16' when autocast is off (the no_autocast word decides that case)."""
    import torch
    try:
        return str(torch.get_autocast_dtype("cuda")).rsplit(".", 1)[-1] if torch.is_autocast_enabled() else "bfloat16"
    except Exception:                                   # pragma: no cover - a torch without get_autocast_dtype
        return "bfloat16"


def _is_oom(e) -> bool:
    try:
        from opt_core.oom import is_oom
        return bool(is_oom(e))
    except Exception:                                   # pragma: no cover - an older core: torch's own class
        import torch
        return isinstance(e, torch.cuda.OutOfMemoryError)


def _error(kind: str, e: Exception, what: str) -> None:
    COUNTS[kind][ERROR_PREFIX + type(e).__name__] += 1
    _PRINTED[kind] += 1
    if _PRINTED[kind] <= 2:
        print("[trunk2_ptx1] %s: kernel error -> the stock statement answers this call: %s: %r" % (what, type(e).__name__, e), file=sys.stderr, flush=True)


def _pf_modules(model):
    """[(name, AttentionPairBias)] of the has_s=False self-attention modules outside the model's DiffusionModule (trunk PairformerStack + confidence
    pairformer); MSA-module / template pair stacks have no AttentionPairBias (c_s = 0). ([], named) when the tree is not the pinned one's."""
    from protenix.model.modules import transformer as T, diffusion as Dm, primitives as P
    dm_ids = set()
    for m in model.modules():
        if isinstance(m, Dm.DiffusionModule):
            dm_ids |= {id(x) for x in m.modules()}
    out, has_s = [], 0
    for name, m in model.named_modules():
        if isinstance(m, T.AttentionPairBias) and id(m) not in dm_ids:
            if getattr(m, "has_s", True) or getattr(m, "cross_attention_mode", False) or not isinstance(getattr(m, "attention", None), P.Attention):
                has_s += 1
                continue
            out.append((name, m))
    if not out:
        return [], "no has_s=False AttentionPairBias modules found outside the DiffusionModule (%d has_s / cross modules seen)" % has_s
    return out, None


def _msa_module(model):
    from protenix.model.modules import pairformer as PF
    mm = [x for x in model.modules() if isinstance(x, PF.MSAModule)]
    return (mm[0], None) if len(mm) == 1 else (None, "expected one MSAModule in the model, found %d" % len(mm))


def _make_pf_smha(apb, cell, key, orig):
    """AttentionPairBias.standard_multihead_attention for one module through the shared core's provider (opt_core.kernels.apb) by TIER word: the
    module boundary's row for this (card, dtype, N) — `composed` = the provider's bias PRODUCER row (pair_bias_planes: LayerNorm(z)+Linear head-major)
    + its CORE row (pair_bias_attention: softmax(q.k*scale + bias) v, sigmoid gate) on stock's own bf16 q/k/v/g projections, then stock's linear_o;
    a boundary row that is the library module itself (`torch_module`), a Refusal by name, or a call outside the envelope takes `orig` (the module's
    own bound statement) for that call, counted by reason — never silent."""
    import torch
    APB = __import__(APB_PROVIDER, fromlist=["_"])
    att = apb.attention
    H, D = int(att.num_heads), int(att.c_hidden)
    cnt, st = COUNTS["pf"], STATE["pf"]
    ln, wz = apb.layernorm_z, apb.linear_nobias_z
    scale = 1.0 / math.sqrt(D)
    tier = tier_word()
    core_cell = APB.cell_word("pf", heads=H, head_dim=D)                      # pf_h16d24 for this model; None = a geometry no cell family lists (the boundary's stock row by name)
    mod_cell = APB.cell_word("module_pf") if (int(getattr(apb, "c_a", 0) or 0), int(getattr(apb, "c_z", 0) or 0)) == (384, 128) else None
    pcache, ccache, memo = {}, {}, {}                                          # the provider's packed-weight caches for this module; memo: boundary Selection per call class

    def _lnb():
        b = getattr(ln, "bias", None)
        if b is None:                                                          # create_offset False: LayerNorm without an offset = a zero offset for the producer rows
            b = memo.get("zero_lnb")
            if b is None or b.device != ln.weight.device:
                b = torch.zeros_like(ln.weight); memo["zero_lnb"] = b
        return b

    def smha(q, kv, z, inplace_safe=False, enable_efficient_fusion=False):
        why = pf_route(tuple(q.shape), kv is q, tuple(z.shape), bool(enable_efficient_fusion), dtype=_dt(q), z_dtype=_dt(z), is_cuda=bool(q.is_cuda),
                       autocast=_autocast_on(), training=bool(apb.training), autocast_dtype=_autocast_dtype())
        if why is not None:                              # step aside by name: counted, the module's own statement answers this call
            cnt[why] += 1
            return orig(q, kv, z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
        N = int(q.shape[-2]); lead = q.shape[:-2]
        try:
            if mod_cell is not None:                     # the MODULE boundary's row by tier word: composed (serve below) | torch_module (= this statement: aside by name) | ...
                mk = ("module", N, _dt(z))
                selm = memo.get(mk)
                if selm is None:
                    selm = APB.select(_cc(), z.dtype, mod_cell, N, word=tier); memo[mk] = selm
                if selm.row != "composed":
                    cnt["stock:module_row=%s" % selm.row] += 1
                    return orig(q, kv, z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
            zz = z.reshape(-1, *z.shape[-3:])[0]
            a2 = q.reshape(-1, N, q.shape[-1]); S = a2.shape[0]
            qh = att.linear_q(a2).view(S, N, H, D); kh = att.linear_k(a2).view(S, N, H, D); vh = att.linear_v(a2).view(S, N, H, D)
            gh = att.linear_g(a2).view(S, N, H, D) if att.gating else None
            bias, selp = APB.pair_bias_planes(zz, ln.weight, _lnb(), wz.weight, word=tier, eps=float(getattr(ln, "eps", 1e-5)), out_layout="hij", cache=pcache)   # [H, N, N] planes: Linear(LayerNorm(z)) head-major, the cell's producer row
            o, selc = APB.pair_bias_attention(qh, kh, vh, bias if bias.dim() == 4 else bias[None], gate=gh, word=tier, layout="snhd", cell=core_cell, scale=scale,
                                              out_dtype=qh.dtype, cache=ccache)   # [S, N, H, D]: the cell's core row (shared bias, sigmoid gate)
            out = att.linear_o(o.reshape(*lead, N, H * D))
        except APB.Refusal as r:                          # a row refused this call BY NAME: printed once per kind, counted, the module's own statement answers the call
            w = "stock:refused:%s" % str(getattr(r, "kind", r)).split(" ")[0][:60]
            cnt[w] += 1
            if _PRINTED[("pf", w)] < 1:
                _PRINTED[("pf", w)] += 1
                print("[trunk2_ptx1] pfattn: provider refusal %s -> the module's own statement for such calls" % (r,), flush=True)
            return orig(q, kv, z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
        except Exception as e:
            if _is_oom(e):
                raise
            _error("pf", e, "pfattn")
            return orig(q, kv, z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
        st["calls"] += 1
        rows = "%s+%s" % (selp.row + ((":" + selp.variant) if getattr(selp, "variant", None) else ""), selc.row + ((":" + selc.variant) if getattr(selc, "variant", None) else ""))
        st["rows"][rows] = st["rows"].get(rows, 0) + 1                         # the served (producer + core) row pair per call, by the provider's words
        cnt[served_word("pf", key)] += 1
        return out

    smha.__wrapped__ = orig; smha._trunk2_ptx1 = "pf"
    return smha


def _make_opm_forward(opm, cell, key, orig):
    """OuterProductMean.forward for one module (triangular/layers.py:724-765 restated around the kernels): eval call, mask None."""
    import torch
    from opt_core.ops.msa_fused.msa_triton import ln_linear, opm_out
    cnt, st, word = COUNTS["opm"], STATE["opm"], served_word("opm", key)
    ln = opm.layer_norm
    eps_ln = float(getattr(ln, "eps", 1e-5))
    cache = {}

    def _fp16() -> bool:
        try:
            from protenix.model.utils import is_fp16_enabled
            return bool(is_fp16_enabled())
        except Exception:
            return False

    def forward(m, mask=None, chunk_size=None, inplace_safe=False):
        why = opm_route(tuple(m.shape), mask is not None, dtype=_dt(m), is_cuda=bool(m.is_cuda), autocast=_autocast_on(), training=bool(opm.training), fp16=_fp16(), autocast_dtype=_autocast_dtype())
        if why is not None:
            cnt[why] += 1
            return orig(m, mask=mask, chunk_size=chunk_size, inplace_safe=inplace_safe)
        try:
            S, N, C = m.shape
            a, b = ln_linear(m if m.is_contiguous() else m.contiguous(), ln.weight, getattr(ln, "bias", None), (opm.linear_1.weight, opm.linear_2.weight),
                             eps_ln, R=int(cell["ln_rows"]))                                          # [S, N, c] bf16 each (stock: linear_k(layer_norm(m)); mask = ones -> unchanged)
            mask1 = m.new_ones(m.shape[:-1]).unsqueeze(-1)                                          # stock :742-747: mask = ones [S, N, 1]; norm = einsum(mask, mask) + eps (the same statement)
            norm = torch.einsum("...abc,...adc->...bdc", mask1, mask1) + opm.eps                     # [N, N, 1]
            at = a.transpose(-2, -3); bt = b.transpose(-2, -3)                                      # [N, S, c] views, as stock
            wt = cache.get("wout_t")
            if wt is None or wt.device != m.device:
                wt = opm.linear_out.weight.detach().t().contiguous().to(torch.bfloat16); cache["wout_t"] = wt
            ob = opm.linear_out.bias
            bias_b = None if ob is None else ob.detach().to(torch.bfloat16)
            cz = wt.shape[1]
            out = torch.empty((N, N, cz), device=m.device, dtype=torch.bfloat16)
            step = N if chunk_size is None else max(1, int(chunk_size))
            for r0 in range(0, N, step):
                r1 = min(N, r0 + step)
                outer = torch.einsum("...bac,...dae->...bdce", at[r0:r1], bt)                         # stock's GEMM statement (cuBLAS), result laid out [b, c, d, e]
                opm_out(outer, wt, bias_b, norm[r0:r1], out[r0:r1], TD=int(cell["td"]), CG=int(cell["cg"]), num_warps=int(cell["num_warps"]), num_stages=int(cell["num_stages"]))
                st["chunks"] += 1
                del outer
        except Exception as e:
            if _is_oom(e):
                raise
            _error("opm", e, "opm_fused")
            return orig(m, mask=mask, chunk_size=chunk_size, inplace_safe=inplace_safe)
        st["calls"] += 1
        cnt[word] += 1
        return out

    forward.__wrapped__ = orig; forward._trunk2_ptx1 = "opm"
    return forward


def _make_pwa_forward(pwa, cell, key, orig):
    """MSAPairWeightedAveraging.forward for one module (pairformer.py:396-422 restated around the kernels)."""
    import torch
    from opt_core.ops.msa_fused.msa_triton import pwa_ln_vg, pwa_out2
    APB = __import__(APB_PROVIDER, fromlist=["_"])
    tier = tier_word()
    cnt, st, word = COUNTS["pwa"], STATE["pwa"], served_word("pwa", key)
    H, CC = int(pwa.n_heads), int(pwa.c)
    ln, lz = pwa.layernorm_m, pwa.layernorm_z
    eps_m, eps_z = float(getattr(ln, "eps", 1e-5)), float(getattr(lz, "eps", 1e-5))
    cache = {}

    def forward(m, z):
        why = pwa_route(tuple(m.shape), tuple(z.shape), dtype=_dt(m), z_dtype=_dt(z), is_cuda=bool(m.is_cuda), autocast=_autocast_on(), training=bool(pwa.training), autocast_dtype=_autocast_dtype())
        if why is not None:
            cnt[why] += 1
            return orig(m, z)
        try:
            v_hm, g = pwa_ln_vg(m if m.is_contiguous() else m.contiguous(), ln.weight, getattr(ln, "bias", None), pwa.linear_no_bias_mv.weight, pwa.linear_no_bias_mg.weight,
                                eps_m, H=H, CC=CC, MB=int(cell["mb"]), NB=int(cell["nb"]), num_warps=int(cell["ln_warps"]))   # v [H, N, M, c] head-major, g logits [M, N, H*c]
            M, N = int(m.shape[0]), int(m.shape[1])
            w = getattr(pwa, "_fpf_zcache_w", None)                     # softmax weights another lever may cache across the chunks of one MSAStack call (no lever of this kit sets it: None)
            if w is None:
                zz = z if (z.stride(2) == 1 and z.stride(0) == N * z.stride(1)) else z.contiguous()
                try:                                                                                # the provider's producer row by tier word: Linear_{c_z->H}(LayerNorm(z)) head-major [H, i, j]
                    lzb = getattr(lz, "bias", None)
                    if lzb is None:
                        lzb = cache.get("zero_lzb")
                        if lzb is None or lzb.device != lz.weight.device:
                            lzb = torch.zeros_like(lz.weight); cache["zero_lzb"] = lzb
                    logits, selz = APB.pair_bias_planes(zz, lz.weight, lzb, pwa.linear_no_bias_z.weight, word=tier, eps=eps_z, out_layout="hij", out_dtype=torch.float32, cache=cache)
                    if logits.dim() == 4:
                        logits = logits[0]
                    st["rows"][selz.row] = st["rows"].get(selz.row, 0) + 1
                except APB.Refusal as r:                                                            # refused by name: the module's own producer statement for this call, counted
                    zr = "zpath_refused:%s" % str(getattr(r, "kind", r)).split(" ")[0][:60]
                    st["rows"][zr] = st["rows"].get(zr, 0) + 1
                    logits = pwa.linear_no_bias_z(lz(zz)).permute(2, 0, 1).float()                     # stock: linear_no_bias_z(layernorm_z(z)) -> heads first
                w8 = torch.softmax(logits, dim=-1).to(torch.bfloat16).contiguous()                 # stock softmax over j (nn.Softmax(dim=-2) on [i, j, h]); bf16 = the einsum's autocast operand dtype
            else:
                w8 = w.permute(2, 0, 1).to(torch.bfloat16).contiguous()
            wv = torch.matmul(w8, v_hm.view(H, N, M * CC)).view(H, N, M, CC)                       # the pair-weighted average: one bmm per head, fp32-accumulated (stock einsum '...ijh,...mjhc->...mihc')
            wt = cache.get("wout_t")
            if wt is None or wt.device != m.device:
                wt = pwa.linear_no_bias_out.weight.detach().t().contiguous().to(torch.bfloat16); cache["wout_t"] = wt
            out = pwa_out2(g, wv, wt, MB=int(cell["out_mb"]), NB=int(cell["out_nb"]), num_warps=int(cell["out_warps"]))   # [M, N, c_m]: linear_out(bf16(sigmoid(g) * wv))
        except Exception as e:
            if _is_oom(e):
                raise
            _error("pwa", e, "pwa_fused")
            return orig(m, z)
        st["calls"] += 1
        cnt[word] += 1
        return out

    forward.__wrapped__ = orig; forward._trunk2_ptx1 = "pwa"
    return forward


def _install(kind: str, model) -> dict:
    st = STATE[kind]
    st["requested"] = True
    if st["installed_on"]:
        return facts(kind)
    try:
        import torch  # noqa: F401
    except Exception as e:                               # pragma: no cover
        st["named"] = "torch not importable: %r" % (e,); return facts(kind)
    cell, key, named = card_cell(kind, _cc())
    st["cell"], st["cell_key"], st["named"] = cell, key, named
    if cell is None:
        return facts(kind)
    try:
        import triton  # noqa: F401
        __import__(APB_PROVIDER, fromlist=["_"])           # pfattn's producer + core rows; pwa_fused's z-path producer rows (the provider, by tier word)
        st["tier"] = tier_word()
        if kind != "pf":
            import opt_core.ops.msa_fused.msa_triton  # noqa: F401
    except Exception as e:
        st["cell"] = None; st["named"] = "kernel package not importable: %r" % (e,); return facts(kind)
    if kind == "pf":
        mods, why = _pf_modules(model)
    else:
        mm, why = _msa_module(model)
        mods = []
        if why is None:
            for i, blk in enumerate(mm.blocks):
                if kind == "opm":
                    sub = getattr(blk, "outer_product_mean_msa", None)
                    if sub is not None:
                        mods.append(("msa_module.blocks.%d.outer_product_mean_msa" % i, sub))
                else:
                    stk = getattr(blk, "msa_stack", None)
                    sub = getattr(stk, "msa_pair_weighted_averaging", None) if stk is not None else None
                    if sub is not None:
                        mods.append(("msa_module.blocks.%d.msa_stack.msa_pair_weighted_averaging" % i, sub))
            if not mods:
                why = "no %s modules under the MSAModule's blocks" % ("OuterProductMean" if kind == "opm" else "MSAPairWeightedAveraging")
    if why:
        st["cell"] = None; st["named"] = why; return facts(kind)
    st["found"] = len(mods)
    skipped = defaultdict(int)
    attr = "standard_multihead_attention" if kind == "pf" else "forward"
    for name, mod in mods:
        if kind == "pf":
            att = mod.attention
            gw = pf_geometry_word(int(att.num_heads), int(att.c_hidden), bool(att.gating), getattr(att.linear_q, "bias", None) is not None, int(mod.linear_nobias_z.in_features))
        elif kind == "opm":
            gw = opm_geometry_word(int(mod.layer_norm.weight.numel()), int(mod.c_hidden), int(mod.linear_out.weight.shape[0]), int(mod.linear_out.weight.shape[1]))
        else:
            gw = pwa_geometry_word(int(mod.layernorm_m.weight.numel()), int(mod.n_heads), int(mod.c), int(mod.linear_no_bias_z.in_features))
        if gw is not None:
            skipped[gw] += 1
            continue
        had = attr in vars(mod)
        prev = vars(mod).get(attr)
        if had and not getattr(prev, "_trunk2_ptx1", None):          # another lever already put an instance-level statement here: left in place and NAMED
            skipped["owned:%s" % getattr(prev, "__name__", type(prev).__name__)] += 1
            continue
        orig = getattr(mod, attr)                        # the bound stock statement
        maker = _make_pf_smha if kind == "pf" else (_make_opm_forward if kind == "opm" else _make_pwa_forward)
        setattr(mod, attr, maker(mod, cell, key, orig))
        _INSTALLED[kind].append((mod, attr, had, prev))
    st["installed_on"] = len(_INSTALLED[kind])
    if skipped:
        words = ", ".join("%s x%d" % kv for kv in sorted(skipped.items()))
        st["named"] = ((st["named"] + "; ") if st["named"] else "") + "not installed on " + words
        for w, n in skipped.items():
            COUNTS[kind]["named:" + w] += n
    return facts(kind)


def install_pfattn(model) -> dict:
    """Install the fused producer + core on the trunk / confidence pairformer AttentionPairBias (has_s=False) modules."""
    return _install("pf", model)


def install_opm(model) -> dict:
    """Install the fused OuterProductMean forward on the MSA module's blocks."""
    return _install("opm", model)


def install_pwa(model) -> dict:
    """Install the fused MSAPairWeightedAveraging forward on the MSA module's MSAStack blocks."""
    return _install("pwa", model)


def apply(model, pfattn: bool = False, opm_fused: bool = False, pwa_fused: bool = False) -> dict:
    """The arm's three words at once (levers_ptx1.set_trunk2): installs what is asked, uninstalls what is not. -> describe()."""
    for word, on in (("pfattn", pfattn), ("opm_fused", opm_fused), ("pwa_fused", pwa_fused)):
        kind = KIND[word]
        if on:
            _install(kind, model)
        else:
            uninstall(model, kind)
            STATE[kind]["requested"] = False
    return describe()


def uninstall(model=None, kind: str = None) -> None:
    """Restore the modules' own statements (kind = pf | opm | pwa | None = all)."""
    for k in (("pf", "opm", "pwa") if kind is None else (kind,)):
        for mod, attr, had, prev in _INSTALLED[k]:
            if had:
                setattr(mod, attr, prev)
            else:
                try:
                    delattr(mod, attr)
                except AttributeError:
                    pass
        _INSTALLED[k] = []
        STATE[k]["installed_on"] = 0


# ---------------------------------------------------------------- account
def facts(kind: str) -> dict:
    """The LEVER facts of one kind: engaged (installed on >= 1 module), installed_on / found, cell key, named condition, Python-level calls, census."""
    st = STATE[kind]
    return {"engaged": bool(st["installed_on"]), "installed_on": st["installed_on"], "found": st["found"], "cell_key": st["cell_key"], "cell": st["cell"],
            "named": st["named"], "calls": st["calls"], "chunks": st["chunks"], "counts": dict(COUNTS[kind]), "packages": package_versions(),
            "tier": st["tier"], "rows": dict(st["rows"])}       # rows: the provider rows served per call (pfattn: producer+core; pwa_fused: its z-path producer), by the provider's words


def package_versions() -> dict:
    """The carried packages' versions as imported in this process (read from sys.modules: the account imports nothing)."""
    return {p: (str(getattr(sys.modules[p], "__version__", "unknown")) if p in sys.modules else "not imported") for p in PACKAGES}


def describe() -> dict:
    return {"version": __version__, "pf": facts("pf"), "opm": facts("opm"), "pwa": facts("pwa")}


def lever_facts(word: str) -> dict:
    """Per arm word, what a LEVER line needs: state on | skipped (+ reason), served / gated / fallback counters in the kit's evidence form. A word whose
    modules made NO call at all in the run (an input without MSA features: MSAModule.forward returns before its blocks) is `aside` — by design, named."""
    kind = KIND[word]
    f = facts(kind)
    counts = f["counts"]
    served = sum(n for w, n in counts.items() if w.startswith(SERVED_PREFIX))
    gated = {w: n for w, n in counts.items() if (w.startswith(ASIDE_PREFIX) or w.startswith("named:")) and n}
    fallback = {w: n for w, n in counts.items() if w.startswith(ERROR_PREFIX) and n}
    on = f["engaged"]
    out = {"state": "on" if on else "skipped", "reason": None if on else (f["named"] or "not_requested"), "served": served, "gated": gated, "fallback": fallback,
           "installed_on": f["installed_on"], "cell": ("%s@%s" % (kind, f["cell_key"])) if on else None, "named": f["named"], "calls": f["calls"]}
    if on and served == 0 and not fallback and not any(w.startswith(ASIDE_PREFIX) for w in gated):
        out["aside"] = "no_call"                          # installed, and the model never entered the site this run (no MSA pass / no trunk pass): by design
    return out
