"""The `big` memory mode's adapter: this kit's memory levers on the shared core's registry (`opt_core.mem`), applied on top of the
mode's composition (modes.py: the base arm's row with the XL lean prologue at every size, no graph pools, no PAD8 provider; the stock
n_token guard lifted — runner_hooks.guard_lift) with the census per item and the fail-closed exit rule through the package's own
partial gate.

The line is one lever set: its levers (`LINE`) and their settings (`SETTINGS`) are constants of the mode — no switch removes or tunes a
lever (a `PROTENIX_V2_BIG_*` variable in the environment is refused by name: _autoload.undeclared); a line lever that cannot
apply makes the mode refuse by name (exit 3: a mode is all of its levers, never a subset under its name). The line (`LINE`, in order) and what each lever does:

    drop_bond_mask   [setting; bitwise]  the featurizer's INT64 [N_atom, N_atom] `bond_mask` (protenix/data/core/featurizer.py
                     `Featurizer.get_mask_features`, the dense np.zeros((num_atoms, num_atoms)) scattered with the ligand-polymer bonds)
                     is replaced by an empty [0, 0] long tensor: nobody reads the key at inference (readers in the pinned tree:
                     protenix/model/loss.py only; tests/test_big.py greps the pinned stock tree and fails on a new reader). Census: the
                     feature dict the trunk receives carries the empty tensor (the `big_seams` post-hook on
                     Protenix.get_pairformer_output reads it) — a full mask on an item is a named fallback.
    cond_chunk       [chunk; band]  DiffusionConditioning.prepare_cache (protenix/model/modules/diffusion.py) row-blocked: the stock body
                     — cat(z_trunk, relpe) -> LayerNorm -> Linear -> two Transitions on the [N, N, 256] pair cache, all per-(i, j)
                     vector, in fp32 under the runner's skip_amp.sample_diffusion (every size: guard_lift keeps it above 3840) — runs on `rows` token rows at a
                     time into a preallocated output. The stock body's pair-sized transients (the cat, the LayerNorm copy, the Transitions'
                     intermediates) shrink to rows/N of their size. Band: a cuBLAS fp32 GEMM at another M may differ by 1 ulp (bf16/autocast
                     sites bitwise); the seed-spread word states the class.
    apb_bias_chunk   [chunk; band]  AttentionPairBias.standard_multihead_attention (protenix/model/modules/transformer.py): the pair
                     bias `linear_nobias_z(layernorm_z(z))` — a full-N LN'd copy of the fp32 pair cache per DiT block per step —
                     computed on `rows` token rows at a time into the [N, N, heads] bias; the attention call is stock's. Only above
                     `above_tok` tokens; the `enable_efficient_fusion` branch (conv2d, no LN copy) is left as stock. Band as cond_chunk.
    cache_release    [allocator; bitwise]  the core's lever, policy per_stage: torch.cuda.empty_cache() at the seams the kit names —
                     after the trunk (Protenix.get_pairformer_output), after the diffusion conditioning cache (prepare_cache), after
                     sampling (before the confidence head) — so the process high-water mark is the largest phase, not the trunk's
                     reserved pool plus the next phase's growth.
    relp_lazy        [chunk; band]  the relative-position one-hot plane is never materialised: RelativePositionEncoding.generate_relp
                     stores a `LazyRelp` (the five [N] token index vectors + the relp geometry) as input_feature_dict["relp"], and every
                     reader gets rows on demand — RelativePositionEncoding.forward (the trunk's z_init term and the diffusion conditioning's
                     relpe term) runs the stock `linear_no_bias` on `rows` token rows of the plane at a time into one preallocated
                     [N, N, c_z] output (the rows are the stock generate_relp statements on a row block, `_relp_rows_fn`), and the row-blocked conditioning (cond_chunk) slices `relp[..., i0:i1, :, :]`, which a
                     LazyRelp serves as those rows. Resident bytes removed: the fp32 [N, N, 139] plane for the whole item. Band: the relpe
                     linear at another M (a bf16 GEMM under autocast) may pick another kernel.
    msa_zfree        [allocator; bitwise]  MSABlock.forward (protenix/model/modules/pairformer.py) re-stated statement by statement with one
                     addition: once `z = z + self.outer_product_mean_msa(...)` has produced the block's new pair tensor, the block-INPUT z's
                     storage is released (no later statement of the pinned inference loop reads it: MSAModule.forward rebinds z per block,
                     Protenix.get_pairformer_output rebinds z to the module's output) — one [N, N, c_z] pair tensor less through each
                     MSA block's pair stack. Refuses by name when MSABlock.forward is owned by another lever at apply time.
    diffcache_free   [allocator; bitwise]  the diffusion conditioning's pair cache (`cache["pair_z"]`, DiffusionConditioning.prepare_cache's
                     [N, N, c] output held by Protenix._main_inference_loop's cache dict until the loop returns) has its storage released
                     when the confidence head starts (ConfidenceHead.forward entry): sampling — every chunk of it (the stock
                     `sample_diffusion_chunk_size` loop) — has consumed it by then and no later statement reads it; one
                     fp32 [N, N, c] tensor off the confidence phase. Refuses by name when ConfidenceHead.forward is owned by another lever.

The chunk levers decide per call through the core's `ChunkPolicy` (engaged above `above_tok` tokens with `rows` rows per block; the
decision is recorded per unit, a threshold is never hidden) and report every chunked call through the core's per-call sink
(`AppliedRecord.call` from `chunk_rows`: the unit's `ran` mark, the event, the per-site label counts).

Not a lever, recorded: `peak` — torch max_memory_allocated / max_memory_reserved at each seam and at the end of each unit, written
as `big_peak.json` beside the outputs (informational; the device's high-water mark is nvidia-smi's).

Hooks resolve lazily: a lever whose target module is not imported yet is armed on a post-import finder (runner_hooks' form) and marks
itself when it patches; a lever whose target never imports is `absent` on the census (partial, fail-closed). Nothing here imports torch
or protenix at module level.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

from . import _core  # noqa: F401  (opt_core importable: installed, else the pinned path)
from . import runner_hooks as _runner_hooks

try:
    from opt_core import mem as _mem
    from opt_core.mem import allocator as _allocator
    from opt_core.mem import registry as _registry
except ImportError as _e:                                 # an older core: every lever refuses by name at activation
    _mem = None
    _allocator = None
    _registry = None
    _IMPORT_ERROR = repr(_e)
else:
    _IMPORT_ERROR = None
_chunk = None                                             # opt_core.mem.chunk imports torch: bound at activation (an activated process), never at import


def _bind_chunk() -> Optional[str]:
    """opt_core.mem.chunk (chunk_rows, ChunkPolicy) bound for the chunk levers; the refusal text when the module is absent."""
    global _chunk
    if _chunk is not None:
        return None
    try:
        from opt_core.mem import chunk as c
    except ImportError as e:
        return f"opt_core.mem.chunk not present in this core ({e!r})"
    _chunk = c
    return None

PREFIX = "PROTENIX_V2"                                    # the kit's variable stem the core records (ctx.prefix); the line reads no PROTENIX_V2_* variable
TAG = "protenix-opt"
LINE = ("drop_bond_mask", "cond_chunk", "apb_bias_chunk", "cache_release", "relp_lazy", "msa_zfree", "diffcache_free")   # the levers of the line, in order (modes.MEM_LEVERS is this tuple)
SETTINGS: Dict[str, Dict[str, Any]] = {                   # the line's settings, constants of the mode (the levers' declared settings on the core registry)
    "cond_chunk": {"rows": 256, "above_tok": 1023},      # DiffusionConditioning.prepare_cache: 256 token rows per block, engaged above 1023 tokens
    "apb_bias_chunk": {"rows": 256, "above_tok": 1023},  # AttentionPairBias pair bias: 256 token rows per block, engaged above 1023 tokens
    "relp_lazy": {"rows": 128},                          # the relative-position plane: 128 token rows per relpe-linear block (every item)
    "cache_release": {"policy": "per_stage"},            # torch.cuda.empty_cache() at the kit's phase seams
}
MARK = "MEM:"                                             # the applied markers stack._classify reads: MEM:<lever>(applied|armed) | MEM:<lever>:refused(...)
PRE_ITEM_UNIT = "pre-item"                                # the census unit before the first item (the model's warm-up forwards): expects no lever
FEATURIZER = "protenix.data.core.featurizer"
DIFFUSION = "protenix.model.modules.diffusion"
TRANSFORMER = "protenix.model.modules.transformer"
MODEL = "protenix.model.protenix"
EMBEDDERS = "protenix.model.modules.embedders"
PAIRFORMER = "protenix.model.modules.pairformer"
CONFIDENCE = "protenix.model.modules.confidence"
RELP_KEYS = ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")   # the token features generate_relp reads (embedders.py RelativePositionEncoding.input_feature)
GIB = float(1 << 30)

_STATE: Dict[str, Any] = {"record": None, "ctx": None, "unit": None, "n_unit": 0, "peaks": [], "seams": {}, "counters": {}, "apb_policy": None,
                          "finders": {}, "patched": {}, "install_error": None, "decisions": {}, "verdict": None,
                          "diffcache": None}               # diffcache_free: a weak reference to the live diffusion pair cache (prepare_cache's output)



def available() -> Optional[str]:
    """None when opt_core.mem is importable, else the refusal text (the named reason every lever carries then)."""
    return None if _mem is not None else f"big: opt_core.mem not present in this core ({_IMPORT_ERROR})"


# ------------------------------------------------------------------------------------------------------------ lazy patching


def _patch_when_imported(lever: str, target: str, apply: Callable[[Any], None]) -> str:
    """Apply now (module imported) or arm a post-import patch; returns 'applied' | 'armed'."""
    mod = sys.modules.get(target)
    if mod is not None:
        apply(mod)
        _STATE["patched"][lever] = True
        return "applied"

    def _apply(module, _lever=lever):
        apply(module)
        _STATE["patched"][_lever] = True
    f = _runner_hooks.PostImportFinder(target, _apply)                       # the kit's one post-import patch form (runner_hooks)
    _STATE["finders"][lever] = f
    sys.meta_path.insert(0, f)
    return "armed"


def _record():
    return _STATE["record"]


def _mark_once(lever: str, detail: Optional[str] = None) -> None:
    """One `ran` event per lever per unit for the levers whose sites are not chunked calls (the chunk levers report every call
    through the core's per-call sink, `record.call`, from `chunk_rows`)."""
    rec = _record()
    if rec is None:
        return
    unit = _STATE["unit"] or _mem.PROCESS_UNIT
    key = (unit, lever)
    _STATE["counters"][key] = _STATE["counters"].get(key, 0) + 1
    if _STATE["counters"][key] == 1:
        rec.mark(lever, detail=detail)


def _skip_once(lever: str, reason: str) -> None:
    """One `skipped` event per lever per unit (a site below its engagement threshold fires on every call of the item)."""
    rec = _record()
    if rec is None:
        return
    unit = _STATE["unit"] or _mem.PROCESS_UNIT
    if unit in rec.units and lever in rec.units[unit].ran:
        return
    key = (unit, lever, "skip")
    _STATE["counters"][key] = _STATE["counters"].get(key, 0) + 1
    if _STATE["counters"][key] == 1:
        rec.skip(lever, reason)


def unit_calls(unit: str, lever: str) -> int:
    """The number of levered calls of `lever` on `unit` (the record's per-call events: one per chunked call)."""
    rec = _record()
    if rec is None or unit not in rec.units:
        return 0
    return sum(1 for e in rec.units[unit].events if e.get("lever") == lever and e.get("kind") == "call")


# ------------------------------------------------------------------------------------------------------------ the levers


def _applies_import(lever: str, target: str):
    def applies(ctx):
        if available():
            return _registry.refuse(lever, "core", available())
        top = target.split(".")[0]
        if target in sys.modules or top in sys.modules:
            return None
        try:
            ok = importlib.util.find_spec(top) is not None
        except (ValueError, ImportError):
            ok = False
        if not ok:
            return _registry.refuse(lever, f"hooks.{target}", f"{top} is not importable in this process")
        return None
    return applies


def _drop_bond_mask_apply(module) -> None:
    cls = getattr(module, "Featurizer")
    orig = cls.get_mask_features
    if getattr(orig, "_big", False):
        return

    def get_mask_features(self, _orig=orig):
        feats = _orig(self)
        bm = feats.get("bond_mask") if isinstance(feats, dict) else None
        if bm is not None:
            import torch
            feats["bond_mask"] = torch.zeros((0, 0), dtype=torch.long)
            _STATE["seams"]["drop_bond_mask_calls"] = _STATE["seams"].get("drop_bond_mask_calls", 0) + 1
        return feats
    get_mask_features._big = True
    get_mask_features.__wrapped__ = orig
    cls.get_mask_features = get_mask_features


def _chunk_policy(lever: str, rows: int, above_tok: int):
    """The core's ChunkPolicy for a chunk lever: engages ABOVE `above_tok` tokens (1 = every item) with `rows` rows per block. The
    core's setting grammar reads 0/off as OFF — a lever whose policy never engages is refused by name at apply, so `above_tok` must
    be a positive token count."""
    pol = _chunk.ChunkPolicy(lever=lever, setting=int(above_tok), rows=int(rows))
    if pol.setting == "off":
        raise _registry.RefusalError(_registry.refuse(lever, "above_tok", f"above_tok={above_tok}: 0/off never engages (the core's setting grammar); "
                                                                          f"a positive token count (1 = every item)"))
    return pol


def _decide(lever: str, policy, n: int):
    """The policy's decision for this call, recorded per unit (the record's events + the block's `decisions`); returns the decision."""
    dec = policy.decide(int(n))
    unit = _STATE["unit"] or _mem.PROCESS_UNIT
    _STATE["decisions"].setdefault(unit, {})[lever] = dec.as_record()
    return dec


COND_REASON = "fp32 GEMMs (the diffusion conditioning under skip_amp at every size) at another M may differ by 1 ulp; bf16 sites bitwise"
APB_REASON = "the fp32 pair-bias linear at another M may differ by 1 ulp; the LayerNorm is per vector"


def _cond_chunk_apply_module(module, rows: int, above_tok: int) -> None:
    cls = getattr(module, "DiffusionConditioning")
    orig = cls.prepare_cache
    if getattr(orig, "_big", False):
        return
    under_diffcache = bool(getattr(orig, "_big_diffcache", False))          # diffcache_free wraps whatever prepare_cache is outermost: install the row-blocked
    if under_diffcache:                                                        # body UNDER its wrapper (it must see the assembled cache, not a row block), then re-wrap
        orig = orig.__wrapped__
    policy = _chunk_policy("cond_chunk", rows, above_tok)

    def prepare_cache(self, relp_feature, z_trunk, inplace_safe=False, _orig=orig):
        n = int(z_trunk.shape[-3])
        dec = _decide("cond_chunk", policy, n)
        if not dec.engaged:
            _skip_once("cond_chunk", dec.detail + ": the stock body")
            return _orig(self, relp_feature, z_trunk, inplace_safe)

        def fn(z_blk, i0, i1):                                              # the stock statement on one row block (cat -> LN -> linear -> 2 Transitions)
            return _orig(self, relp_feature[..., i0:i1, :, :], z_blk, inplace_safe)
        out = _chunk.chunk_rows(fn, z_trunk, -3, dec.rows, exact="band", reason=COND_REASON, record=_record(), with_offsets=True,
                                lever="cond_chunk", site="prepare_cache", extra={"policy": dec.source, "threshold": dec.threshold})
        _seam("cond_cache")
        return out
    prepare_cache._big = True
    prepare_cache.__wrapped__ = orig
    cls.prepare_cache = prepare_cache
    if under_diffcache:
        _diffcache_free_apply_diffusion(module)


def _apb_bias_chunk_apply_module(module, rows: int, above_tok: int) -> None:
    cls = getattr(module, "AttentionPairBias")
    orig = cls.standard_multihead_attention
    if getattr(orig, "_big", False):
        return
    pfd = getattr(module, "permute_final_dims")
    policy = _chunk_policy("apb_bias_chunk", rows, above_tok)
    _STATE["apb_policy"] = policy                                                # the live policy, for a binding that owns the method on an INSTANCE (apb_core's pf_attn: apb_bias_chunked)

    def standard_multihead_attention(self, q, kv, z, inplace_safe=False, enable_efficient_fusion=False, _orig=orig):
        n = int(z.shape[-3])
        if enable_efficient_fusion:
            _skip_once("apb_bias_chunk", "efficient-fusion branch (stock: the LayerNorm folded into a conv weight; no full-N LN copy)")
            return _orig(self, q, kv, z, inplace_safe, enable_efficient_fusion)
        dec = _decide("apb_bias_chunk", policy, n)
        if not dec.engaged:
            _skip_once("apb_bias_chunk", dec.detail + ": the stock body")
            return _orig(self, q, kv, z, inplace_safe, enable_efficient_fusion)

        def fn(z_blk):                                                        # the stock statement per row block: linear_nobias_z(layernorm_z(z))
            return self.linear_nobias_z(self.layernorm_z(z_blk))
        bias = _chunk.chunk_rows(fn, z, -3, dec.rows, exact="band", reason=APB_REASON, record=_record(), lever="apb_bias_chunk", site="pair_bias",
                                 extra={"policy": dec.source, "threshold": dec.threshold})
        bias = pfd(bias, [2, 0, 1])                                             # [..., n_heads, N_token, N_token]
        return self.attention(q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe)
    standard_multihead_attention._big = True
    standard_multihead_attention.__wrapped__ = orig
    cls.standard_multihead_attention = standard_multihead_attention



def apb_bias_chunked(apb_module, z):
    """The apb_bias_chunk lever's chunked pair-bias statement for a binding that owns ``standard_multihead_attention`` on an INSTANCE of
    AttentionPairBias (apb_core's pf_attn, whose instance method the class-level re-statement above does not reach):
    when the lever is applied in this process and its policy engages for z's token count, the stock bias ``linear_nobias_z(layernorm_z(z))``
    is computed on ``rows`` token rows at a time (the same statement, policy words and census events as the class-level re-statement: the
    unit counts under ``units_ran`` / ``calls``) and returned permuted to [..., n_heads, N_token, N_token]; None when the lever is not
    applied here or the policy does not engage at this size (named once per unit) -- the binding then produces the bias its own way."""
    policy = _STATE.get("apb_policy")
    if policy is None or _chunk is None:
        return None
    n = int(z.shape[-3])
    dec = _decide("apb_bias_chunk", policy, n)
    if not dec.engaged:
        _skip_once("apb_bias_chunk", dec.detail + ": the binding's producer")
        return None

    def fn(z_blk):                                                            # the stock statement per row block: linear_nobias_z(layernorm_z(z))
        return apb_module.linear_nobias_z(apb_module.layernorm_z(z_blk))
    bias = _chunk.chunk_rows(fn, z, -3, dec.rows, exact="band", reason=APB_REASON, record=_record(), lever="apb_bias_chunk", site="pair_bias",
                             extra={"policy": dec.source, "threshold": dec.threshold})
    d = bias.dim()
    return bias.permute(*range(d - 3), d - 1, d - 3, d - 2)                     # [..., n_heads, N_token, N_token] (permute_final_dims(bias, [2, 0, 1]))



class _ZRows:
    """Carries the pair representation from a wrapped ``layernorm_z`` to its ``linear_nobias_z`` (the two are one statement,
    ``linear_nobias_z(layernorm_z(z))``, at every AttentionPairBias pair-bias site: the stock method, the DiT hoist's per-block producer and
    the fused DiT stack's per-block producer) so the pair is evaluated on row blocks by ``apb_bias_chunked``."""
    __slots__ = ("z",)

    def __init__(self, z):
        self.z = z


def _wrap_pair_bias_modules(apb_module) -> bool:
    """Instance-level forwards on ONE AttentionPairBias module's ``layernorm_z`` / ``linear_nobias_z``: above the gate ``layernorm_z(z)`` hands z
    on (no full-N normalized copy) and ``linear_nobias_z`` answers with the row-blocked statement (``apb_bias_chunked``, un-permuted: the
    caller permutes as stock does); below the gate / a LayerNorm output consumed elsewhere both run their own forward. Idempotent."""
    ln, lin = apb_module.layernorm_z, apb_module.linear_nobias_z
    if getattr(ln.__dict__.get("forward"), "_big", False):
        return False
    o_ln, o_lin = ln.forward, lin.forward

    def ln_fwd(z, _o=o_ln):
        policy = _STATE.get("apb_policy")
        if policy is None or _chunk is None or not hasattr(z, "dim") or z.dim() < 3 or not policy.decide(int(z.shape[-3])).engaged:
            return _o(z)
        return _ZRows(z)

    def lin_fwd(x, _o=o_lin, _m=apb_module):
        if not isinstance(x, _ZRows):
            return _o(x)
        b = apb_bias_chunked(_PairSite(o_ln, o_lin), x.z)                          # [.., H, N, N]; the stock modules' own forwards per row block
        if b is None:                                                             # the policy stopped engaging between the two calls (never in one statement): the stock pair
            return _o(o_ln(x.z))
        d = b.dim()
        return b.permute(*range(d - 3), d - 2, d - 1, d - 3)                       # back to [.., N, N, H]: the caller's permute_final_dims([2, 0, 1]) restores [.., H, N, N]
    ln_fwd._big = True; lin_fwd._big = True
    ln_fwd.__wrapped__ = o_ln; lin_fwd.__wrapped__ = o_lin
    ln.forward = ln_fwd; lin.forward = lin_fwd
    return True


class _PairSite:
    """The two stock callables of a wrapped module, in the shape ``apb_bias_chunked`` reads (``.layernorm_z`` / ``.linear_nobias_z``)."""
    __slots__ = ("layernorm_z", "linear_nobias_z")

    def __init__(self, ln, lin):
        self.layernorm_z, self.linear_nobias_z = ln, lin


def _install_dit_pair_bias_rows(runner) -> None:
    """Runner-seam installer (registered when apb_bias_chunk applies): wrap the pair-bias LayerNorm / projection pair of the DiffusionModule's
    24 token AttentionPairBias blocks, whose ``standard_multihead_attention`` the DiT hoist and the fused DiT stack own at the instance level
    (their producers evaluate ``linear_nobias_z(layernorm_z(z))`` themselves, so the class-level re-statement never reaches them). Names the
    result on the kit's stream; a model without that topology is named and left alone (the class-level re-statement still serves stock modules)."""
    try:
        from protenix.model.modules import diffusion as Dm
        model = runner.model
        dms = [model] if isinstance(model, Dm.DiffusionModule) else [m for m in model.modules() if isinstance(m, Dm.DiffusionModule)]
        n = 0
        for dm in dms:
            for blk in dm.diffusion_transformer.blocks:
                n += int(_wrap_pair_bias_modules(blk.attention_pair_bias))
        _STATE["dit_pair_sites"] = _STATE.get("dit_pair_sites", 0) + n
        stacks = dit_bias_per_block(model)                                      # the fused DiT stack, if it is installed already (sampler_levers binds it after its own install otherwise)
        sys.stderr.write(f"[{TAG}] MEM:apb_bias_chunk(dit_sites={n} dit_stacks_per_block={stacks}; the DiffusionTransformer blocks' pair-bias LayerNorm + projection on "
                         f"{SETTINGS['apb_bias_chunk']['rows']}-row blocks above {SETTINGS['apb_bias_chunk']['above_tok']} tokens on every sampler route; the fused stack's "
                         f"block biases produced one block at a time)\n")
    except Exception as e:  # noqa: BLE001 -- named, never fatal: the class-level re-statement and the bindings' producers still ask the line per call
        sys.stderr.write(f"[{TAG}] MEM:apb_bias_chunk(dit_sites=0; {e!r})\n")



def znorm_permuted_rows(ln_forward, z):
    """big: the DiffusionModule's ``normalize(z)`` (LayerNorm over the pair channels) followed by stock's ``permute_final_dims(z, [2, 0, 1])
    .contiguous()`` produced as ONE channel-major [.., C, N, N] tensor, the LayerNorm evaluated on ``rows`` token rows at a time straight into it --
    instead of the full normalized copy [.., N, N, C] AND its permuted copy alive together (two pair-sized fp32 tensors at the sampler's per-call
    normalize).  LayerNorm statistics are per (i, j) over the channels, so the row-blocked evaluation is the full
    one bitwise; the strided write is a copy.  Engaged when the memory line is active (apb_bias_chunk applied) above its token gate; None otherwise
    (the caller runs the stock two-step statement).  Returns the contiguous channel-major tensor (the caller hands on its [.., N, N, C] view)."""
    policy = _STATE.get("apb_policy")
    if policy is None or _chunk is None or not hasattr(z, "dim") or z.dim() < 3:
        return None
    n = int(z.shape[-3])
    dec = policy.decide(n)
    if not dec.engaged:
        return None
    import torch
    rows = max(1, int(dec.rows))
    lead = tuple(z.shape[:-3]); N1, N2, C = int(z.shape[-3]), int(z.shape[-2]), None
    out = None
    for i0 in range(0, N1, rows):
        blk = ln_forward(z[..., i0:i0 + rows, :, :])                              # [.., r, N, C]: the module's own LayerNorm on a row block
        if out is None:
            C = int(blk.shape[-1])
            out = torch.empty(lead + (C, N1, N2), dtype=blk.dtype, device=blk.device)   # channel-major, contiguous: what stock's permute(...).contiguous() allocates
        d = blk.dim()
        out[..., :, i0:i0 + blk.shape[-3], :].copy_(blk.permute(*range(d - 3), d - 1, d - 3, d - 2))
    _STATE["znorm_rows_calls"] = _STATE.get("znorm_rows_calls", 0) + 1
    return out



class _LazyBias:
    """A DiT block's pair bias not yet produced (the fused DiT stack asks for all 24 before its block loop; big produces each at its block)."""
    __slots__ = ("i", "z", "eff")

    def __init__(self, i, z, eff):
        self.i, self.z, self.eff = i, z, eff


class _ApbPerBlock:
    """The fused stack's attention module seen through big: ``dit_apb`` produces a lazy block bias right before the block's attention (the
    stack's own ``_bias``: the hoist protocol when a hoist entry is live, else the producer -- row-blocked above apb_bias_chunk's gate) and lets
    it go after the call; everything else is the module itself."""

    def __init__(self, real, stack, bias_fn):
        self._real, self._stack, self._bias_fn = real, stack, bias_fn

    def __getattr__(self, name):
        return getattr(self._real, name)

    def dit_apb(self, qkvg, bias, *a, **k):
        if isinstance(bias, _LazyBias):
            _STATE["dit_bias_per_block"] = _STATE.get("dit_bias_per_block", 0) + 1
            bias = self._bias_fn(self._stack, bias.i, bias.z, bias.eff)          # ONE block's [.., 16, N, N] planes live at a time (24 before: 23 x 16*N*N*4 B held across the stack)
        return self._real.dit_apb(qkvg, bias, *a, **k)


def dit_bias_per_block(model) -> int:
    """big: the fused DiT stack (third_party/protenix_fpf_ditfast, lever dit_fused) evaluates its 24 blocks' pair biases one block at a time
    instead of all 24 before the block loop -- on the sampler routes without a live hoist entry (the eager / stock routes above the admission
    ceiling) those are 24 resident fp32 [16, N, N] planes per denoiser call; here one block's is live at a time.  Same producers, same
    kernels, same per-block order of producer -> attention: the values are the ones the stack computes.  Applied from the outside on the
    stack INSTANCE (``_bias`` hands a lazy handle on, the attention module is seen through ``_ApbPerBlock``); the stack's bytes are unchanged.
    Returns the number of stacks bound (0: dit_fused not installed on this model)."""
    n = 0
    try:
        from protenix.model.modules import diffusion as Dm
        dms = [model] if isinstance(model, Dm.DiffusionModule) else [m for m in model.modules() if isinstance(m, Dm.DiffusionModule)]
    except Exception:  # noqa: BLE001
        dms = [getattr(model, "diffusion_module", model)]
    for dm in dms:
        dt = getattr(dm, "diffusion_transformer", None)
        st = getattr(dt, "_protenix_fpf_ditfast", None) if dt is not None else None
        if st is None or isinstance(getattr(st, "_apb", None), _ApbPerBlock):
            continue
        bias_fn = type(st)._bias                                               # the stack's own per-block producer (hoist protocol / local cache / producer)
        st._apb = _ApbPerBlock(st._apb, st, bias_fn)
        st._bias = lambda i, z, eff, _L=_LazyBias: _L(i, z, eff)               # instance attribute: the block loop's up-front list holds handles, not planes
        n += 1
    _STATE["dit_stacks_per_block"] = _STATE.get("dit_stacks_per_block", 0) + n
    return n



def _relp_channels(r_max: int, s_max: int) -> int:
    """The relp channel count (embedders.py RelativePositionEncoding: 4·r_max + 2·s_max + 7 = 139 at the stock r_max=32, s_max=2)."""
    return 4 * int(r_max) + 2 * int(s_max) + 7


def _relp_carrier(feats, r_max: int, s_max: int):
    """A zero-stride [N, N, C] view over the first token feature: the chunk axis chunk_rows walks (the plane's rank and extent, no memory)."""
    n = int(feats[0].shape[-1])
    return feats[0].view(n, 1, 1).expand(n, n, _relp_channels(r_max, s_max))


def _relp_rows_fn(feats, r_max: int, s_max: int) -> Callable:
    """The stock generate_relp statements (protenix/model/modules/embedders.py RelativePositionEncoding.generate_relp) on token rows
    [i0, i1): the left operand of every `[..., :, None] - [..., None, :]` pair sliced to the rows, the right operand whole; returns the
    plane's rows as fp32 [i1 - i0, N, C] — per-(i, j) integer arithmetic and one-hot then `.float()`, exact per element; relp_lazy
    serves rows on demand from it (the plane never held). `feats` = the five 1-D [N] token features in RELP_KEYS order."""
    torch = sys.modules["torch"]
    F = torch.nn.functional
    asym_id, residue_index, entity_id, token_index, sym_id = feats
    r_max, s_max = int(r_max), int(s_max)

    def fn(blk, i0, i1):
        a_i, r_i, e_i, t_i, s_i = (f[i0:i1] for f in feats)
        b_same_chain = (a_i[:, None] == asym_id[None, :]).long()
        b_same_residue = (r_i[:, None] == residue_index[None, :]).long()
        b_same_entity = (e_i[:, None] == entity_id[None, :]).long()
        d_residue = torch.clip(input=r_i[:, None] - residue_index[None, :] + r_max, min=0, max=2 * r_max) * b_same_chain + (1 - b_same_chain) * (2 * r_max + 1)
        a_rel_pos = F.one_hot(d_residue, 2 * (r_max + 1))
        d_token = torch.clip(input=t_i[:, None] - token_index[None, :] + r_max, min=0, max=2 * r_max) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (2 * r_max + 1)
        a_rel_token = F.one_hot(d_token, 2 * (r_max + 1))
        d_chain = torch.clip(input=s_i[:, None] - sym_id[None, :] + s_max, min=0, max=2 * s_max) * b_same_entity + (1 - b_same_entity) * (2 * s_max + 1)
        a_rel_chain = F.one_hot(d_chain, 2 * (s_max + 1))
        return torch.cat([a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain], dim=-1).float()
    return fn


class LazyRelp:
    """input_feature_dict["relp"] under relp_lazy: the five [N] token index vectors (RELP_KEYS) and the relp geometry — never the
    [N, N, C] one-hot plane. `rows(i0, i1)` returns the stock plane's rows [i0, i1) (fp32 [i1 - i0, N, C]: `_relp_rows_fn`, the stock
    statements on a row block); `self[..., i0:i1, :, :]` returns the same rows (the slice the row-blocked conditioning, cond_chunk, takes
    per block); any other indexing raises TypeError by name (a reader this lever does not serve is a defect to name, never a silent dense
    rebuild). `shape` / `dtype` / `device` / `dim()` / `size()` / `numel()` / `element_size()` describe the plane it stands for; `to()`
    changes nothing (the index vectors live where the item's features live)."""

    __slots__ = ("feats", "r_max", "s_max", "n", "c", "shape", "dtype", "device", "is_cuda", "_fn")

    def __init__(self, feats, r_max: int, s_max: int):
        torch = sys.modules["torch"]
        self.feats = tuple(feats)
        self.r_max, self.s_max = int(r_max), int(s_max)
        self.n = int(self.feats[0].shape[-1])
        self.c = _relp_channels(self.r_max, self.s_max)
        self.shape = (self.n, self.n, self.c)
        self.dtype = torch.float32
        self.device = self.feats[0].device
        self.is_cuda = bool(self.feats[0].is_cuda)
        self._fn = _relp_rows_fn(self.feats, self.r_max, self.s_max)

    def rows(self, i0: int, i1: int):
        torch = sys.modules["torch"]
        i0, i1 = max(0, int(i0)), min(self.n, int(i1))
        with torch.no_grad():
            return self._fn(None, i0, i1)

    def carrier(self):
        return _relp_carrier(self.feats, self.r_max, self.s_max)

    def __getitem__(self, key):
        if (isinstance(key, tuple) and len(key) == 4 and key[0] is Ellipsis and isinstance(key[1], slice) and key[1].step in (None, 1)
                and key[2] == slice(None) and key[3] == slice(None)):
            i0 = 0 if key[1].start is None else int(key[1].start)
            i1 = self.n if key[1].stop is None else int(key[1].stop)
            return self.rows(i0, i1)
        raise TypeError(f"LazyRelp: unsupported index {key!r} — the plane is never materialised; rows(i0, i1) and [..., i0:i1, :, :] are served")

    def dim(self) -> int:
        return 3

    def size(self, i=None):
        return self.shape if i is None else self.shape[i]

    def numel(self) -> int:
        return self.n * self.n * self.c

    nelement = numel

    def element_size(self) -> int:
        return 4

    def to(self, *a, **k):
        return self

    def __repr__(self) -> str:
        return f"LazyRelp(n_token={self.n}, channels={self.c}, device={self.device})"


LAZY_REASON = ("the one-hot rows are the stock generate_relp integer statements on a row block (exact per element); the relpe linear_no_bias runs per row "
               "block — a GEMM at another M may pick another kernel (1-ulp class; bf16 under the trunk's autocast, fp32 under the conditioning's skip_amp)")
MSA_ZFREE_REASON = ("the stock MSABlock.forward statements in the stock order on the same tensors; the one addition releases the storage of the block-input z "
                    "after its last read (the outer-product-mean residual add) — arithmetic untouched")
DIFFCACHE_REASON = ("a storage release of the diffusion pair cache after its last read (sampling has ended when ConfidenceHead.forward starts; the confidence "
                    "head reads z_trunk, not the cache) — arithmetic untouched")


def _owner(fn) -> str:
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', repr(fn)))}"


def _stock_owned(fn, module_name: str, qualname: str) -> bool:
    """True when `fn` is the class's own function (its __qualname__ ends with `qualname`, nothing wrapped, no lever marker), i.e. no
    other lever has replaced or wrapped the site; a foreign owner (another module's replacement function) is refused by the caller BY NAME."""
    q = getattr(fn, "__qualname__", "") or ""
    return q.endswith(qualname) and not hasattr(fn, "__wrapped__") and not getattr(fn, "_big", False)


def _relp_lazy_apply_module(module, rows: int) -> None:
    """RelativePositionEncoding (stock protenix/model/modules/embedders.py): generate_relp stores a LazyRelp instead of the fp32 [N, N, C]
    plane; forward on a LazyRelp runs the stock `self.linear_no_bias(relp_feature)` per block of `rows` token rows into one preallocated
    [N, N, c_z] output (chunk_rows, exact word band: LAZY_REASON); forward on a tensor (a dense plane, or the row block cond_chunk's
    per-block prepare_cache passes) is the stock statement unchanged."""
    cls = getattr(module, "RelativePositionEncoding")
    orig_gen, orig_fwd = cls.generate_relp, cls.forward
    if getattr(orig_gen, "_big_lazy", False):
        return
    if getattr(orig_gen, "_big", False) or not _stock_owned(orig_gen, EMBEDDERS, "RelativePositionEncoding.generate_relp"):
        raise _registry.RefusalError(_registry.refuse("relp_lazy", "site_owned", f"RelativePositionEncoding.generate_relp is owned by {_owner(orig_gen)}: relp_lazy "
                                                                                f"replaces the stock body and would drop it (the mode refuses by name)"))
    if not _stock_owned(orig_fwd, EMBEDDERS, "RelativePositionEncoding.forward"):
        raise _registry.RefusalError(_registry.refuse("relp_lazy", "site_owned", f"RelativePositionEncoding.forward is owned by {_owner(orig_fwd)}"))
    torch = sys.modules["torch"]

    def generate_relp(self, input_feature_dict, _orig=orig_gen):
        feats = tuple(input_feature_dict[k] for k in RELP_KEYS)
        if any(f.dim() != 1 for f in feats):
            _skip_once("relp_lazy", f"batched token features (dims {[int(f.dim()) for f in feats]}): the stock body")
            return _orig(self, input_feature_dict)
        input_feature_dict["relp"] = LazyRelp(feats, self.r_max, self.s_max)     # the rows are built under no_grad on demand (the stock body is a no_grad block too)
        _STATE["seams"]["relp_lazy_items"] = _STATE["seams"].get("relp_lazy_items", 0) + 1
        _seam("relp")
        return input_feature_dict

    def forward(self, relp_feature, _orig=orig_fwd):
        if not isinstance(relp_feature, LazyRelp):
            return _orig(self, relp_feature)                                   # a tensor: the stock statement (a dense plane, or one row block of it from cond_chunk)

        def fn(blk, i0, i1):                                                    # the stock forward statement on the plane's rows [i0, i1) (ambient autocast decides the dtype, as stock)
            return _orig(self, relp_feature.rows(i0, i1))
        return _chunk.chunk_rows(fn, relp_feature.carrier(), 0, int(rows), exact="band", reason=LAZY_REASON, record=_record(), with_offsets=True,
                                 lever="relp_lazy", site="relpe_linear", extra={"n_token": relp_feature.n, "c_z": int(getattr(self, "c_z", 0) or 0)})
    generate_relp._big = True
    generate_relp._big_lazy = True
    generate_relp.__wrapped__ = orig_gen
    forward._big = True
    forward.__wrapped__ = orig_fwd
    cls.generate_relp = generate_relp
    cls.forward = forward


def _msa_zfree_apply_module(module) -> None:
    """MSABlock.forward (stock protenix/model/modules/pairformer.py, protenix 2.0.0) re-stated statement by statement; the addition releases
    the block-input z's storage once the outer-product-mean residual add has produced the block's new z (tests/test_big.py locks the
    re-statement to the pinned stock source). Refused by name when the site is owned by another lever."""
    cls = getattr(module, "MSABlock")
    orig = cls.forward
    if getattr(orig, "_big", False):
        return
    if not _stock_owned(orig, PAIRFORMER, "MSABlock.forward"):
        raise _registry.RefusalError(_registry.refuse("msa_zfree", "site_owned", f"MSABlock.forward is owned by {_owner(orig)}: msa_zfree re-states the stock body and "
                                                                                f"would drop it (the mode refuses by name)"))
    torch = sys.modules["torch"]

    def forward(self, m, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None, _orig=orig):
        z_in = z
        # Communication
        z = z + self.outer_product_mean_msa(
            m, inplace_safe=inplace_safe, chunk_size=chunk_size
        )
        if torch.is_grad_enabled():                                             # the addition: the block-input pair tensor's storage released after its last read
            _skip_once("msa_zfree", "autograd enabled (a training-form call): nothing released")
        elif z_in.untyped_storage().data_ptr() == z.untyped_storage().data_ptr():
            _skip_once("msa_zfree", "the residual add returned the input's storage: nothing to release")
        else:
            st = z_in.untyped_storage()
            nb = int(st.nbytes())
            st.resize_(0)                                                       # read by no later statement (MSAModule.forward / get_pairformer_output rebind z)
            _STATE["seams"]["msa_zfree_bytes_max"] = max(_STATE["seams"].get("msa_zfree_bytes_max", 0), nb)
            _STATE["seams"]["msa_zfree_calls"] = _STATE["seams"].get("msa_zfree_calls", 0) + 1
            _mark_once("msa_zfree", detail=f"block-input z storage released ({nb} B)")
        del z_in
        if not self.is_last_block:
            # MSA stack
            m = self.msa_stack(m, z)
        # Pair stack
        _, z = self.pair_stack(
            s=None,
            z=z,
            pair_mask=pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        if not self.is_last_block:
            return m, z
        else:
            return None, z  # to ensure that `m` will not be used.
    forward._big = True
    forward.__wrapped__ = orig
    cls.forward = forward


def _diffcache_free_apply_diffusion(module) -> None:
    """DiffusionConditioning.prepare_cache wrapped (whatever owns it — the stock body or cond_chunk's row-blocked one — is CALLED, never
    replaced): the output, the diffusion pair cache, is remembered by weak reference for the confidence-entry release."""
    cls = getattr(module, "DiffusionConditioning")
    orig = cls.prepare_cache
    if getattr(orig, "_big_diffcache", False):
        return
    torch = sys.modules["torch"]
    import weakref

    def prepare_cache(self, relp_feature, z_trunk, inplace_safe=False, _orig=orig):
        out = _orig(self, relp_feature, z_trunk, inplace_safe)
        if torch.is_tensor(out) and not torch.is_grad_enabled():
            _STATE["diffcache"] = weakref.ref(out)
        return out
    prepare_cache._big_diffcache = True
    prepare_cache._big = getattr(orig, "_big", False)                      # cond_chunk's marker carried: its own idempotence check reads it
    prepare_cache.__wrapped__ = orig
    cls.prepare_cache = prepare_cache


def _diffcache_free_apply_confidence(module) -> None:
    """ConfidenceHead.forward wrapped: on entry the remembered diffusion pair cache's storage is released (sampling has ended; the cache
    dict of Protenix._main_inference_loop still holds the tensor, nothing reads it again). Refused by name when the site is owned by
    another lever."""
    cls = getattr(module, "ConfidenceHead")
    orig = cls.forward
    if getattr(orig, "_big", False):
        return
    if not _stock_owned(orig, CONFIDENCE, "ConfidenceHead.forward"):
        raise _registry.RefusalError(_registry.refuse("diffcache_free", "site_owned", f"ConfidenceHead.forward is owned by {_owner(orig)} "
                                                                                     f"(the mode refuses by name)"))

    def forward(self, *a, _orig=orig, **k):
        ref = _STATE.get("diffcache")
        t = ref() if ref is not None else None
        _STATE["diffcache"] = None
        if t is None:
            _skip_once("diffcache_free", "no diffusion pair cache alive at the confidence head (shared-variable cache off, or already released)")
        else:
            st = t.untyped_storage()
            nb = int(st.nbytes())
            if nb > 0:
                st.resize_(0)
                _STATE["seams"]["diffcache_freed_bytes"] = _STATE["seams"].get("diffcache_freed_bytes", 0) + nb
                _mark_once("diffcache_free", detail=f"diffusion pair cache storage released ({nb} B)")
            else:
                _skip_once("diffcache_free", "the diffusion pair cache's storage was already empty")
            del t
        return _orig(self, *a, **k)
    forward._big = True
    forward.__wrapped__ = orig
    cls.forward = forward


def _register_levers() -> None:
    if _registry is None:
        return
    if "cond_chunk" in _registry.LEVERS:
        return

    @_registry.register("drop_bond_mask", family="setting", exact="bitwise",
                        exact_reason="the INT64 atom-pair bond mask is unread at inference (readers: protenix/model/loss.py only)",
                        applies=_applies_import("drop_bond_mask", FEATURIZER),
                        description="the featurizer's dense [N_atom, N_atom] int64 bond_mask replaced by an empty tensor",
                        preconditions=("core", f"hooks.{FEATURIZER}"), settings=())
    def drop_bond_mask(ctx):
        how = _patch_when_imported("drop_bond_mask", FEATURIZER, _drop_bond_mask_apply)
        return _registry.Applied(lever="drop_bond_mask", settings={"placeholder": "[0, 0] int64"},
                                 sites=(f"{FEATURIZER}.Featurizer.get_mask_features",), notes=[how])

    @_registry.register("cond_chunk", family="chunk", exact="band",
                        exact_reason="fp32 GEMMs (the diffusion conditioning under skip_amp) at another M may differ by 1 ulp; bf16 sites bitwise",
                        applies=_applies_import("cond_chunk", DIFFUSION),
                        description="DiffusionConditioning.prepare_cache row-blocked (the stock body per block of `rows` token rows)",
                        preconditions=("core", f"hooks.{DIFFUSION}"), settings=("rows", "above_tok"))
    def cond_chunk(ctx):
        miss = _bind_chunk()
        if miss:
            raise _registry.RefusalError(_registry.refuse("cond_chunk", "core", miss))
        rows = ctx.setting("cond_chunk", "rows", SETTINGS["cond_chunk"]["rows"], cast=int)
        above_tok = ctx.setting("cond_chunk", "above_tok", SETTINGS["cond_chunk"]["above_tok"], cast=int)      # engages ABOVE above_tok tokens (ChunkPolicy); 1 = every item; 0/off refused
        if rows < 1:
            raise _registry.RefusalError(_registry.refuse("cond_chunk", "rows", f"rows={rows}: a positive row count"))
        _chunk_policy("cond_chunk", rows, above_tok)                            # the setting refused by name here, at apply, not at the first item
        how = _patch_when_imported("cond_chunk", DIFFUSION, lambda m: _cond_chunk_apply_module(m, rows, above_tok))
        return _registry.Applied(lever="cond_chunk", settings={"rows": rows, "above_tok": above_tok},
                                 sites=(f"{DIFFUSION}.DiffusionConditioning.prepare_cache",), notes=[how])

    @_registry.register("apb_bias_chunk", family="chunk", exact="band",
                        exact_reason="the fp32 pair-bias linear at another M may differ by 1 ulp; the LayerNorm is per vector",
                        applies=_applies_import("apb_bias_chunk", TRANSFORMER),
                        description="AttentionPairBias pair bias linear_nobias_z(layernorm_z(z)) computed on `rows` token rows at a time",
                        preconditions=("core", f"hooks.{TRANSFORMER}"), settings=("rows", "above_tok"))
    def apb_bias_chunk(ctx):
        miss = _bind_chunk()
        if miss:
            raise _registry.RefusalError(_registry.refuse("apb_bias_chunk", "core", miss))
        rows = ctx.setting("apb_bias_chunk", "rows", SETTINGS["apb_bias_chunk"]["rows"], cast=int)
        above_tok = ctx.setting("apb_bias_chunk", "above_tok", SETTINGS["apb_bias_chunk"]["above_tok"], cast=int)  # engages ABOVE above_tok tokens (ChunkPolicy); 1 = every item; 0/off refused
        if rows < 1:
            raise _registry.RefusalError(_registry.refuse("apb_bias_chunk", "rows", f"rows={rows}: a positive row count"))
        _chunk_policy("apb_bias_chunk", rows, above_tok)
        how = _patch_when_imported("apb_bias_chunk", TRANSFORMER, lambda m: _apb_bias_chunk_apply_module(m, rows, above_tok))
        from . import runner_seam                                                   # + the DiT blocks' own pair-bias modules once the runner's model exists (the hoist / fused-stack producers)
        runner_seam.add("apb_bias_chunk", _install_dit_pair_bias_rows); runner_seam.arm()
        return _registry.Applied(lever="apb_bias_chunk", settings={"rows": rows, "above_tok": above_tok},
                                 sites=(f"{TRANSFORMER}.AttentionPairBias.standard_multihead_attention",), notes=[how])

    @_registry.register("relp_lazy", family="chunk", exact="band",
                        exact_reason=LAZY_REASON, applies=_applies_import("relp_lazy", EMBEDDERS),
                        description="the relative-position one-hot plane never materialised: a LazyRelp in input_feature_dict['relp'], the relpe linear per block of `rows` token rows, cond_chunk served row slices",
                        preconditions=("core", f"hooks.{EMBEDDERS}", "site_owned"), settings=("rows",))
    def relp_lazy(ctx):
        miss = _bind_chunk()
        if miss:
            raise _registry.RefusalError(_registry.refuse("relp_lazy", "core", miss))
        rows = ctx.setting("relp_lazy", "rows", SETTINGS["relp_lazy"]["rows"], cast=int)   # token rows of the plane per relpe-linear block (every item: the plane is never built, no size gate)
        if rows < 1:
            raise _registry.RefusalError(_registry.refuse("relp_lazy", "rows", f"rows={rows}: a positive row count"))
        how = _patch_when_imported("relp_lazy", EMBEDDERS, lambda m: _relp_lazy_apply_module(m, rows))
        return _registry.Applied(lever="relp_lazy", settings={"rows": rows},
                                 sites=(f"{EMBEDDERS}.RelativePositionEncoding.generate_relp", f"{EMBEDDERS}.RelativePositionEncoding.forward"), notes=[how])

    @_registry.register("msa_zfree", family="allocator", exact="bitwise",
                        exact_reason=MSA_ZFREE_REASON, applies=_applies_import("msa_zfree", PAIRFORMER),
                        description="MSABlock.forward re-stated with the block-input pair tensor's storage released after the outer-product-mean residual add",
                        preconditions=("core", f"hooks.{PAIRFORMER}", "site_owned"), settings=())
    def msa_zfree(ctx):
        how = _patch_when_imported("msa_zfree", PAIRFORMER, _msa_zfree_apply_module)
        return _registry.Applied(lever="msa_zfree", settings={}, sites=(f"{PAIRFORMER}.MSABlock.forward",), notes=[how])

    @_registry.register("diffcache_free", family="allocator", exact="bitwise",
                        exact_reason=DIFFCACHE_REASON, applies=_applies_import("diffcache_free", DIFFUSION),
                        description="the diffusion conditioning pair cache's storage released when the confidence head starts (after sampling, its last reader)",
                        preconditions=("core", f"hooks.{DIFFUSION}", f"hooks.{CONFIDENCE}", "site_owned"), settings=())
    def diffcache_free(ctx):
        how = _patch_when_imported("diffcache_free", DIFFUSION, _diffcache_free_apply_diffusion)
        how_c = _patch_when_imported("diffcache_free:confidence", CONFIDENCE, _diffcache_free_apply_confidence)
        return _registry.Applied(lever="diffcache_free", settings={},
                                 sites=(f"{DIFFUSION}.DiffusionConditioning.prepare_cache", f"{CONFIDENCE}.ConfidenceHead.forward"), notes=[how, f"confidence:{how_c}"])


# ------------------------------------------------------------------------------------------------------------ seams, units, peak


def _counters() -> dict:
    if _allocator is None:
        return {}
    try:
        return _allocator.counters()
    except Exception as e:                                  # noqa: BLE001
        return {"error": repr(e)}


def _seam_counters() -> dict:
    """The allocator counters under seam_* names (the adapter's attribution columns, never the table's peak columns)."""
    return {("seam_" + k if k != "device" else k): v for k, v in _counters().items()}


def _seam(point: str) -> None:
    """A phase seam: the peak counters recorded, the core's cache release at the stage (policy per_stage)."""
    ctx = _STATE["ctx"]
    _STATE["peaks"].append({"unit": _STATE["unit"], "seam": point, "t": time.time(), **_seam_counters()})
    if ctx is not None:
        _allocator.release(ctx, "stage")


def _seams_apply(module) -> None:
    cls = getattr(module, "Protenix")
    orig = cls.get_pairformer_output
    if getattr(orig, "_big", False):
        return

    def get_pairformer_output(self, input_feature_dict, *a, _orig=orig, **k):
        out = _orig(self, input_feature_dict, *a, **k)
        bm = input_feature_dict.get("bond_mask") if isinstance(input_feature_dict, dict) else None
        rec = _record()
        if rec is not None and "drop_bond_mask" in rec.expected:
            if bm is None:
                rec.skip("drop_bond_mask", "no bond_mask feature on the item")
            elif int(bm.numel()) == 0:
                _mark_once("drop_bond_mask", detail="bond_mask empty on the trunk's input")
            else:
                rec.fallback("drop_bond_mask", f"bond_mask present with shape {tuple(bm.shape)}: the featurizer ran un-levered "
                                               "(a dataloader worker forked before the patch?)")
        _seam("trunk_end")
        return out
    get_pairformer_output._big = True
    get_pairformer_output.__wrapped__ = orig
    cls.get_pairformer_output = get_pairformer_output
    if hasattr(cls, "sample_diffusion"):
        orig_sd = cls.sample_diffusion

        def sample_diffusion(self, *a, _orig=orig_sd, **k):
            out = _orig(self, *a, **k)
            _seam("sampling_end")
            return out
        sample_diffusion._big = True
        sample_diffusion.__wrapped__ = orig_sd
        cls.sample_diffusion = sample_diffusion


def _on_item(n_token: int) -> None:
    """runner_hooks' per-item listener (update_inference_configs, once per (item, seed)): the census unit."""
    rec = _record()
    if rec is None:
        return
    if _STATE["unit"] is not None:
        _unit_end()
    _STATE["n_unit"] += 1
    unit = f"item{_STATE['n_unit']}:{int(n_token)}tok"
    _STATE["unit"] = unit
    rec.unit_begin(unit)
    if _allocator is not None:
        _allocator.reset_peak()
    _STATE["peaks"].append({"unit": unit, "seam": "unit_begin", "t": time.time(), **_seam_counters()})


def _unit_end() -> None:
    rec = _record()
    unit = _STATE["unit"]
    if rec is None or unit is None:
        return
    _STATE["peaks"].append({"unit": unit, "seam": "unit_end", "t": time.time(), **_seam_counters()})
    if unit == PRE_ITEM_UNIT:
        rec.unit_end(unit)
        _STATE["unit"] = None
        return
    for lever in ("cond_chunk", "apb_bias_chunk"):                         # a chunk lever whose site never ran on the unit: the trunk-only item (named)
        if lever in rec.expected and lever not in rec.units[unit].ran and lever not in rec.units[unit].skipped:
            rec.skip(lever, "site not reached on this unit")
    if "cache_release" in rec.expected and "cache_release" not in rec.units[unit].ran:
        rec.skip("cache_release", "no seam reached on this unit")
    patched = _STATE["patched"]
    for lever, keys in (("relp_lazy", ("relp_lazy",)), ("msa_zfree", ("msa_zfree",)), ("diffcache_free", ("diffcache_free", "diffcache_free:confidence"))):
        if (lever in rec.expected and lever not in rec.units[unit].ran and lever not in rec.units[unit].skipped
                and all(patched.get(k) for k in keys)):                          # the patch is live and the item ended before its site: named; a target that never imported stays absent (partial, fail-closed)
            rec.skip(lever, "site not reached on this unit")
    rec.unit_end(unit)
    _STATE["unit"] = None


# ------------------------------------------------------------------------------------------------------------ activation


def activate(base: str, environ=None) -> List[str]:
    """Apply the line on top of the composed mode (called by stack._apply for mode big, after the kit's levers). Returns the
    applied markers for stack._classify: `MEM:<lever>(applied|armed)` per applied lever, `MEM:<lever>:refused(<precondition>)` per
    refusal (a refused line lever is a fallback of the row: the package's partial gate exits 3 by name). The core reads
    nothing from ``environ`` for the line (the allocator lever's mechanism only): the levers and their settings are passed."""
    environ = os.environ if environ is None else environ
    why = available()
    if why:
        _STATE["install_error"] = why
        return [f"{MARK}{lv}:refused(core)" for lv in LINE]
    _register_levers()
    ctx = _registry.Ctx(prefix=PREFIX, tag=TAG, framework="torch", environ=environ, graphs=False,
                        settings={"cache_release": dict(SETTINGS["cache_release"])},
                        hooks={lv: {"module": m} for lv, m in (("drop_bond_mask", FEATURIZER), ("cond_chunk", DIFFUSION),
                                                                ("apb_bias_chunk", TRANSFORMER), ("cache_release", MODEL),
                                                                ("relp_lazy", EMBEDDERS), ("msa_zfree", PAIRFORMER), ("diffcache_free", DIFFUSION))},
                        extra={"base": base, "line": list(LINE), "seams": ["relp", "trunk_end", "cond_cache", "sampling_end"]})
    line = _mem.BigLine(base=base, levers=LINE, lines={"default": {"levers": tuple(LINE), "tier": "band", "cost_note": "the default line"}})   # the one declared line
    rec = _mem.apply(line, ctx, strict=False, switches=None, allow_partial=False)   # the line as declared, no switch, no opt-out: a refused lever is a fallback of the row and the package's partial gate exits 3
    _STATE["record"], _STATE["ctx"] = rec, ctx
    rec.unit_begin(PRE_ITEM_UNIT, expected=())                                 # what runs before the first item (warm-up forwards) is accounted, expected nothing
    _STATE["unit"] = PRE_ITEM_UNIT
    _patch_when_imported("big_seams", MODEL, _seams_apply)
    _runner_hooks.install()                                                      # the per-item hook armed (idempotent; guard_lift arms it too): the census delimiter
    _runner_hooks.add_item_listener(_on_item)
    marks = []
    for a in rec.applied:
        marks.append(f"{MARK}{a.lever}({a.notes[0] if a.notes else 'applied'})")
    for r in rec.refused:
        marks.append(f"{MARK}{r.lever}:refused({r.precondition})")
    return marks


def state() -> Optional[dict]:
    """The record's manifest block (kit.big) with the seams and the peak summary, or None when the mode never activated here."""
    rec = _record()
    if rec is None:
        return None if _STATE["install_error"] is None else {"refused": _STATE["install_error"], "line": list(LINE)}
    if _STATE["unit"] is not None:
        _unit_end()
    block = rec.manifest_block()
    if _STATE.get("verdict") is not None:
        block["exit"] = dict(_STATE["verdict"])                                  # the kit's verdict (the core's, or fail-closed when no item unit was observed)
    block["seams"] = dict(_STATE["seams"])
    block["decisions"] = {u: dict(d) for u, d in _STATE["decisions"].items()}     # the chunk policies' per-unit decisions (threshold never hidden)
    block["patched"] = dict(_STATE["patched"])
    return block


def _token(v) -> str:
    """A LEVER-line value is one token (opt_core.report.lever_line refuses blanks): whitespace runs become `_`."""
    return "_".join(str(v).split())


def evidence(name: str, block: Optional[dict]) -> list:
    """The LEVER line's evidence pairs of a memory lever from the mode's block (state()): its exactness label, its settings, and the
    census counts — units where its site ran / was skipped by name / fell back, and its levered calls (the record's per-call events)."""
    if not block or block.get("refused"):
        return [("refused", 1)] if block else []
    lv = next((a for a in (block.get("levers") or []) if a.get("lever") == name), None)
    if lv is None:
        return []
    pairs = [("exact", lv.get("exact"))] + [(k, _token(v)) for k, v in sorted((lv.get("settings") or {}).items())]
    units = ((block.get("census") or {}).get("units") or {})
    ran = sum(1 for u in units.values() if name in (u.get("ran") or []))
    skipped = sum(1 for u in units.values() if name in (u.get("skipped") or {}))
    fell = sum(1 for u in units.values() if name in (u.get("fallback") or {}))
    calls = sum(n for site in ((lv.get("detail") or {}).get("calls") or {}).values() for n in site.values())
    if name == "apb_bias_chunk":                                                 # the DiT sites wrapped from the outside + the fused stack's per-block biases
        pairs = pairs + [("dit_sites", int(_STATE.get("dit_pair_sites", 0))), ("dit_stacks_per_block", int(_STATE.get("dit_stacks_per_block", 0))),
                         ("dit_bias_per_block", int(_STATE.get("dit_bias_per_block", 0))), ("znorm_rows_calls", int(_STATE.get("znorm_rows_calls", 0)))]
    return pairs + [("units_ran", ran), ("units_skipped", skipped), ("units_fallback", fell), ("calls", calls or ran)]


def reconcile(rep: dict) -> dict:
    """The kit's report brought up to date with the census: a line lever that refused, fell back or never marked on a unit joins
    `levers_fallback` with its reason (the package's partial gate turns it into exit 3);
    the block rides the report as `big`."""
    rec = _record()
    if rec is None:
        if _STATE["install_error"] is not None:
            rep = dict(rep)
            rep["big"] = {"refused": _STATE["install_error"], "line": list(LINE)}
        return rep
    if _STATE["unit"] is not None:
        _unit_end()
    v = rec.exit_gate(0, expect_units=True, allow_partial=False)   # the core's verdict on the record (no opt-out): its partial list names the fallbacks the package's gate refuses on
    if _STATE["n_unit"] == 0:                                                # no item unit ever opened (the runner's per-item hook never fired): fail-closed by name
        v = dict(v, partial=list(rec.expected), reasons={lv: "no item unit observed (runner_hooks' per-item hook never fired)" for lv in rec.expected},
                 exit_code=v.get("exit_code") if not rec.expected else 3, no_item_unit=True)
    _STATE["verdict"] = v
    r = dict(rep)
    on, fb = list(r.get("levers_applied") or []), list(r.get("levers_fallback") or [])
    why = dict(r.get("fallback_reasons") or {})
    for lever in v["partial"]:
        if lever in on:
            on.remove(lever)
        if lever not in fb:
            fb.append(lever)
        why[lever] = "big: " + str(v["reasons"].get(lever, "partial"))
    r.update(levers_applied=on, levers_fallback=fb, fallback_reasons=why, partial=bool(fb), levers_unavailable=list(fb))
    r["big"] = state()
    return r
