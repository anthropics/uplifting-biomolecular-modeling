"""The mode table — the one registry of lever switches in this tree (``configs/*.env`` carry deployment parameters only).

A mode is a list of levers (``registry.LEVERS``) with its activation row — the environment the kits' worker runs under — and its
activation order; nothing here re-implements a lever (stack.py stages the kit files and launches the worker with the row; the levers
read their switches themselves).

    off    stock — no environment set, no lever applied (the stock CLI in a clean subprocess: stock_pred.py)
    exact  byte-identical to the stock as shipped (cuEquivariance kernels on, production defaults): the persistent worker under
           _EXACT_ROW, kernels on — every lever of the row is bitwise = stock (the engine adapters attach them: worker_launch --attach)
    fast   exact's composition plus the Tier-2 cells (_FAST_ROW), kernels on: numerics within the Tier-2 band vs the shipped stock
           (judged against the stock's own seed-to-seed band per target)
    big  the memory mode (Tier 2, composed on fast): fast's trunk and pair-track levers with the providers' ``big`` tier words, no
           CUDA graphs (the sampler roll-out, the DiT hoist and the trunk graphs leave the row by name), the memory line (boltz2_opt.big
           over opt_core.mem: _XL_ROW) and the allocator's expandable segments; lower peak GPU memory than the stock's, slower than fast

Ablation is row-level: a row's ``off`` entries take levers off BY NAME (``resolve``: ``take_off``) — ``off=<lever>:<reason>`` on the
ACTIVE line, ``LEVER name=<lever> state=off reason=<reason>``; no environment word, no flag.
Routes: the kits' line is the persistent worker (``PINNED_ROUTE``); ``enable()`` in a stock-CLI process and the ``BOLTZ2_OPT`` env route
refuse ``exact`` / ``fast`` by name (``CLI_ROUTE``: this tree carries no hook that chains the levers in one ``boltz predict`` process).
One activation row for every GPU key. The user-facing modes are exactly ``MODE_NAMES``; any other name is refused by name.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .registry import DIT as _DIT, TRUNK as _TRUNK

MODE_NAMES = ("off", "exact", "fast", "big")     # the user-facing modes, named by guarantee: stock | bit-identical to the shipped stock | Tier-2 band | the memory mode
DEFAULT_MODE = "fast"                    # the package default: fast wherever a fast mode ships (a --mode-less call); BOLTZ2_OPT unset = nothing applied
PINNED_ROUTE = "worker"

# The activation rows: the trunk levers' switch (BOLTZ_LEVERS), the flash triangle attention's (BOLTZ_TRIATTN=flash BOLTZ_TRIATTN_MIN_TOKENS=300),
# the sampler's (BOLTZ_SAMPLER_ROLLOUT=graph BOLTZ_DIT_HOIST=2) and the engine adapters' words below. --kernels off|on is the worker's own switch (bz_worker_lev.py:31).
# Every lever takes its switch from ONE row word (registry.LEVERS switch / value; a comma-list word carries one unit per lever); the rows are the
# only place a word is set (stack.child_env strips every switch family from the caller's environment first: stock/PINS.json must_be_absent_prefixes).
_SAMPLER_ROW = {"BOLTZ_SAMPLER_ROLLOUT": "graph", "BOLTZ_DIT_HOIST": "2"}   # the boundary-graph roll-out of AtomDiffusion.sample (boltz2_opt.sampler over opt/forward/sampler: lever rollout —
                                                                             # ONE CUDA graph per step boundary, RNG pre-drawn in stock order, SVD/det eager between replays; bitwise) + the DiT hoist
                                                                             # (opt/forward/dit_hoist, level 2: its cache is read inside the captured boundary). The roll-out REPLACES the kit's per-step
                                                                             # graph patch (lever graph_sampler, BOLTZ_GRAPH_DIFFUSION=graph: an available word in no row; both own AtomDiffusion.sample and
                                                                             # the adapter refuses the pair by name) [SAMPLER]
_DITEXACT_ROW = {"BOLTZ_DIT_EXACT": "par,mask,sba,smx,glue"} # the EXACT row only: the token transformer's parallel-branch schedule (dit_par), the all-ones token-mask skip (dit_mask), the fused
                                                        # scale+bias(+mask) pass with ATen's roundings (dit_sba) and the fused softmax replica (dit_smx, bit-compared per class at run time) inside the captured step, on the hoist's level-2 cache (boltz2_opt.ditexact over
                                                        # opt/forward/ditexact/src/dx_token.py); bitwise. The fast row's token transformer is the sampler's fused bf16 step (dit_fused) [DITEXACT]
_GRAPH_TRUNK_ROW = {"BOLTZ_GRAPH_TRUNK": "pf,pfnoseq,templ", "BOLTZ_GRAPH_TRUNK_MAX_TOKENS": "300"}   # the trunk CUDA-graph lever (boltz2_opt.graph over opt/forward/graph/boltz_graph_trunk.py):
                                                        # capture/replay of the Pairformer stacks (trunk + confidence), the template pair stack per item shape + the template boundaries hoisted +
                                                        # the s-track overlap, eager above 300 tokens (first-of-shape break-even; GPU-bound above) — Tier 1: a replay re-issues the
                                                        # captured kernels [GRAPH]
# The exact row (kernels ON — the shipped stock's own triangle path): the roll-out and hoist with the trunk levers that act beside upstream's fused
# kernels (boltz_trunk_levers.py: resid = _pf_layer_forward / _pf_noseq_layer_forward :93-129, mask2 = the AttentionPairBias patch _apb_forward
# :169-191; the fused triangle kernels themselves run upstream's code). The worker runs `--kernels on` (bz_worker_lev.py:31,89,99).
# The engine adapters compose the exact-class cells onto it, each byte-identical to the stock statement it replaces (attached by worker_launch after
# boltz's own module imports; no worker bytes edited): the exact TriMul, the fused transition, the fused attention block, the ATen layer_norm
# replica, the DITEXACT schedule, the MSA module's chunk hoists, the exact fused dim-64 transition, the atom-attention gather / glue hoists, the
# trunk CUDA graphs.
_EXACT_ONLY = ("BOLTZ_DIT_EXACT", "BOLTZ_TRIATTN_EXACT", "BOLTZ_EXACTLN_RESID")   # BOLTZ_TRIATTN_EXACT: fast / big bind the triangle-attention provider's tolerance-class words through pairblock's own variants; BOLTZ_EXACTLN_RESID: fast / big keep z bf16-resident under the PAIRFUSE layer driver (no fp32 residual stream on the C=128 stacks: the fuser's sites are the trunk levers' layer forwards, which the driver replaces) — the exact row's words the fast row does not inherit: fast's token transformer is the fused bf16 step (dit_fused), its C=128 pair stacks the K2B core under PAIRFUSE
_EXACT_ROW = dict({"BOLTZ_LEVERS": "resid,mask2", **_SAMPLER_ROW, "BOLTZ_SAMPLER_ALIGN": "aligncap", **_DITEXACT_ROW},   # BOLTZ_SAMPLER_ALIGN=aligncap: the roll-out's rigid-alignment SVD through WASTE's bitwise cusolver gesvd seam (lever align_aligncap, Tier 1: same LAPACK call, no clone / host read between replays); the fast row takes the in-graph Kabsch instead
                  BOLTZ_FPF_TRIMUL="exact", BOLTZ_FPF_TRIMUL_PROVIDER="exact",                  # the TriMul by the core provider's `exact` TIER word at every N and c_z (boltz2_opt.trimul over opt_core.trimul.by_word / opt_core.kernels.trimul: per call an exact-class kernel row the provider vouches byte-identical on this stack serves through its face; where the provider names the library op as the cell's exact row the engine's own call of it serves, by name — no kernel, cell table or token gate of this tree's)
                  BOLTZ_TRANSITION="exact",                                       # the core transition provider's `exact` TIER word around the module's own LayerNorm on the unchunked bf16-autocast calls (boltz2_opt.transition over opt_core.kernels.transition: its bitwise arm where its table vouches this stack and size, else the module forward by name)
                  BOLTZ_PAIRBLOCK="cueq",                                          # the core's fused triangle-attention prologue/epilogue around the STOCK cuEquivariance attention core (boltz2_opt.pairblock over opt_core.attn.pair_fused.tri_attn_block)
                  BOLTZ_TRIATTN_EXACT="1",                                       # + that attention core asked of the core's triangle-attention provider by its `exact` tier word (lever triattn_exact, boltz2_opt.triattn_exact): its vouched exact-class kernel row on this (card, torch, library) stack, the library call itself everywhere else, per cell and by name
                  BOLTZ_TEMPL_SKIP="1",                                            # the dummy-template elision (lever templ_skip): a pass whose template mask has no slot set gets the module's update from its output projection alone (u_proj(relu(0)) — the module's own statement on the zero aggregation) without the input projections, the Pairformer and v_norm; templated passes run it as stock
                  BOLTZ_EXACTLN="core",                                             # the bitwise replica of ATen's layer_norm forward on every large LayerNorm call of the worker, the pair-track adapters' LayerNorm->bf16 cast fused into its store (boltz2_opt.exactln over forward/exactln) [EXACTLN]
                  BOLTZ_EXACTLN_RESID="on",                                        # + the pair residual add fused with the NEXT block's LayerNorm in one pass over z (lever exactln_resid: forward/exactln exactln_resid_fwd behind the trunk levers' RESID_FUSER hook, boltz_trunk_levers._fused; torch's promote-add bits + the replica's LayerNorm bits, first call per class bit-compared; from 2**26 elements of z) [EXACTLN E4]
                  **_GRAPH_TRUNK_ROW,
                  BOLTZ_WASTE="chunkcast,opmmask,opmdiv,ctorskip",                 # the MSA module's stock chunked paths (> 384 tokens) with the per-chunk bf16 casts applied once per call, OuterProductMean's num_mask computed once per mask tensor and its chunk result divided straight into a contiguous buffer; the model constructor's overwritten init draws reduced to their RNG draw (boltz2_opt.waste over forward/waste/boltz_waste_levers.py: verbatim stock bodies + the hoists; bitwise) [WASTE]
                  BOLTZ_MSA2="pwa2x,trans2x",                                            # the MSA module's PairWeightedAveraging (pwa2x, lever msa_pwa_exact) and the dim-64 / hidden-256 Transitions (trans2x, lever msa_trans2_exact) as fused Triton cells in stock-rounding mode (boltz2_opt.msa2 over forward/msa2): exact under the LOCK DISCIPLINE — below a size floor the stock statements serve by name (small_class), above it every call is bit-compared against the stock statements and returns the STOCK output until >= 1e6 output elements compared equal (a selecting compare locks a summation structure only when exactly one distinct candidate matches), first mismatch refuses the class by name [MSA]
                  BOLTZ_ATOM="keys,glue",
                  BOLTZ_WRITER="overlap",                                        # boltz's writer + the per-seed move off the predicting thread: pinned D2H staging behind one event, one writer process of the zygote's lineage (lever writer_overlap, Tier 1, every row)
                  BOLTZ_PREFETCH="persistent")                                   # the next input featurized by ONE long-lived helper spawned through the zygote instead of a loader worker forked from the CUDA process per input (lever prefetch, Tier 1, both rows); LAST word of the row)                                          # the atom-attention exact units (boltz2_opt.atom over opt/forward/atom: single_to_keys one-hot einsum -> gather; atom<->token casts/mean hoisted per sample(), decoder one-hot bmm -> row gather) — bitwise [ATOM]
# The fast row composes on the exact row (kernels on): the same attachments with their Tier-2 variants — the fused triangle-attention block takes
# the core's flash triangle attention as its attention core at >= 300 tokens (boltz2_opt.pairblock: cuEquivariance's core below the gate, inside
# the same fused prologue/epilogue), the fused transition also serves the MSA module's hidden-chunked calls unchunked, and the TriMul lever binds
# the core provider's `fast` tier word (boltz2_opt.trimul: the measured tolerance-class winner per cell, the template Pairformer's C=64 TriMul included)
# where the exact row binds its `exact` word; boltz2_opt.msa_kernels puts the carried bundle's fused Triton kernels in place of the MSA module's
# outer-product-mean and pair-weighted averaging (every call: Boltz-2 builds both of the kernels' pinned widths only); and the PAIRFUSE
# layer driver takes every C=128 pair stack (trunk, MSA-module pair layers, confidence) on ONE resident bf16 z per stack call over those same cores
# (the block adapters serve the stacks it hands back and say so: superseded_by=pairfuse@c128); the roll-out's token transformer is the fused bf16
# step, the atom encoder/decoder the fused Triton kernels at bf16 operands, the dim-64 transitions the fused kernel in its fast rounding.
# NUMERICS CLASS of the fast row's precision levers (reduced precision where upstream runs fp32; every one its own word): pairfuse (bf16-resident pair
# representation between sub-layers), dit_fused (bf16 DiT token transformer), atom_gemm (bf16 atom-attention operands); flash_triattn / fpf_trimul /
# fpf_opm / fpf_pwa / msa_trans2 / condproj / atom_fused / fused_transition(fast) change reduction order or rounding placement at stock's operand precision.
_FAST_ROW = dict({k: v for k, v in _EXACT_ROW.items() if k not in _EXACT_ONLY and k not in ("BOLTZ_WRITER", "BOLTZ_PREFETCH")},
                 BOLTZ_PAIRBLOCK="default",      # the same fused block with its OWN keyed attention core from 300 tokens (pairblock variant `default`:
                                                                                  # opt_core.attn.pair_fused core="default" — the triattn provider's `fast` tier word at and above its table's
                                                                                  # keys threshold, the flash_triattn cell below it); cuEquivariance's core below the gate
                 BOLTZ_PAIRBLOCK_C64="1",                                          # + the template Pairformer's C=64 tri-attention stack through the same block (lever pairblock_c64: opt_core's qualified (64,4,32) cells, tolerance class; cuEquivariance's core below pairblock.C64_CORE_MIN_TOKENS=800, K2B at and above)
                 BOLTZ_TRANSITION="fast",                                        # + the hidden-chunked MSA-module transitions served unchunked (Tier 2 there)
                 BOLTZ_FPF_TRIMUL="1", BOLTZ_FPF_TRIMUL_PROVIDER="fast",         # the TriMul by the core provider's `fast` TIER word (tolerance class; the measured winner per cell) on the stacks the PAIRFUSE driver hands back (the template Pairformer's C=64 TriMul, inputs below its floor); the big row names `big`
                 BOLTZ_FPF_MSA="opm,pwa",                                        # + the carried bundle's fused MSA-module Triton kernels on every OuterProductMean / PairWeightedAveraging call (boltz2_opt.msa_kernels over fpf_msa.boltz2; Tier 2)
                 **_GRAPH_TRUNK_ROW,                                             # the trunk CUDA-graph lever with all three units: under the PAIRFUSE driver PairformerModule / PairformerNoSeqModule are captured with the DRIVER's bodies (graph BODIES table: pairfuse.pfm_forward / pfnm_forward, digests pinned; a changed body refuses the unit by name at apply), the template module's as stock [GRAPH]
                 BOLTZ_PAIRFUSE="bf16,trimul=core.fast,triatt=core.fast,transition=core.fast",   # the TriangleAttention site by the core triangle-attention provider's `fast` TIER word at every served token count (opt_core.kernels.triattn: the measured cell per (cc, dtype, head_dim, heads, N) decides the row on each card; no kit row pick, no kit token floor); the PAIRFUSE layer driver; its TriMul site by the core provider's `fast` TIER word (the cell table names row native on 9.0 / v4 on 8.0 at c_z 128 — the big row names `core.big`); (boltz2_opt.pairfuse over opt/forward/pairfuse): every C=128 pair stack on ONE resident bf16 z per call — LN in the cores' prologues, residual in their epilogues in place, ending node by strides, seq pair bias in one pass; PRECISION lever (bf16 pair residency; fp32 refused by name until released); the three block adapters above serve the stacks it hands back (templates C=64, kernels off) [PAIRFUSE]
                 BOLTZ_SAMPLER_ALIGN="jacobi64",                                # the rigid alignment's 3x3 SVD + det as ONE sync-free in-graph kernel (lever align_jacobi64, fp64 one-sided Jacobi; Tier 2 outside the network): no host sync left in the sampling loop
                 BOLTZ_SAMPLER_DIT="bf16",                                       # the roll-out's fused bf16 token-transformer step (lever dit_fused, forward/sampler/src/bz_sampler_dit.py; PRECISION lever: bf16 where the 24 DiT layers run fp32 'highest') [SAMPLER]
                 BOLTZ_MSA2="trans2",                                            # the fused dim-64 transition kernel in its fast rounding (lever msa_trans2; Tier 2) [MSA]
                 BOLTZ_CONF="condproj",                                          # DiffusionConditioning's 24+3+3 LayerNorm+Linear pair-bias projections as one shared-statistics normalisation + one folded GEMM per list (boltz2_opt.conf over opt/forward/conf/cond_levers.py; Tier 2) [CONF]
                 BOLTZ_ATOM="keys,fused", BOLTZ_ATOM_GEMM="bf16",
                 BOLTZ_WRITER=_EXACT_ROW["BOLTZ_WRITER"], BOLTZ_PREFETCH="persistent")               # the fused atom encoder/decoder kernels (boltz2_opt.atom: 8+8 launches per denoiser step; lever atom_fused) at bf16 dot operands / fp32 accumulate (lever atom_gemm, PRECISION), the keys gather beside [ATOM]
_F2_ROW = {"BOLTZ_TRIATTN": "flash", "BOLTZ_TRIATTN_MIN_TOKENS": "300"}          # the flash triangle-attention patch (boltz_flash_triattn_patch.py: the flash core inside the STOCK surround) — the memory line's attention lever
_TRIMUL_ROW = {k: _FAST_ROW[k] for k in ("BOLTZ_FPF_TRIMUL", "BOLTZ_FPF_TRIMUL_PROVIDER")}
# AVAILABLE words in no mode row (registry levers with their attachment and evidence rule wired; a row may name them): BOLTZ_PRECISION=dit_tf32|dit_bf16|dit_attn_bf16|seq_bf16 (dit_tf32: TF32 for the score model's remaining fp32 GEMMs — bank-clean, an available word until its same-session forward gain is shown on top of dit_fused / atom_fused;
## (the bf16-autocast score model with its fp32 pins — reference implementation of the policy dit_fused/atom_fused are held to —, the token-attention SDPA core,
# the sequence-attention bf16 unit: bank-clean, off on the timing numbers; `off:<unit>` = a named ablation entry on the same word) [PRECISION], BOLTZ_GRAPH_DIFFUSION=graph
# (graph_sampler: the per-step graph patch the roll-out replaces), BOLTZ_GRAPH_TRUNK+=sovl (graph_trunk's layer-overlap unit: races the in-place
# resid lever at small N — out of the rows until fixed), BOLTZ_CONF=tfeat|tdummy (parked: tdummy is input-conditional),
# BOLTZ_MSA2=hoist (msa_hoist is not a registry lever: the WASTE words own the exact MSA-module hoists). Retired:
# BOLTZ_TRIMUL_SITE / BOLTZ_TRIMUL_FP8 (trimul_site / trimul_fp8, the fused TriMul site on the in-tree bz2_trimul kit copy —
# its kernels are the core's provider row `bz2`, opt_core.kernels.trimul; the PAIRFUSE picks trimul=site|site_tanh refuse by name).
# The memory rows: the engine adapter's line (boltz2_opt.big on the core's memory-mode library opt_core.mem; registry entries xl_trans /
# xl_cond / xl_free / relpos_lazy / expandable_segments = big.BIG_LEVERS) — the row-chunked pair Transition and diffusion conditioner at 256
# rows from the first token (BOLTZ_XL_MIN_TOKENS=0: the levers act on every panel bin, so no row is vacuous), the pre-confidence
# free, the lazily released / recomputed relative-position encoding, and the allocator's expandable segments for the worker process, on the eager
# sampler without the DiT hoist (the graph pool and the hoist's L2 cache are resident memory: ~20 GB at 2565 tokens). BOLTZ_XL_LEVERS names the
# unit levers by token (big.XL_NAMES). `big` composes on the fast row's trunk levers and fused pair track (BIG_ATTN_BASE, the providers'
# `big` tier words), kernels on (Tier 2); the sampler roll-out's whole-loop graph, the DiT hoist and the trunk CUDA graphs leave the row by
# name (MODES["big"]["off"]: no CUDA graphs in the memory mode), the levers that ride them with them (NEEDS).
_XL_ROW = {"BOLTZ_XL": "1", "BOLTZ_XL_LEVERS": "trans,cond,free,relpos", "BOLTZ_XL_MIN_TOKENS": "0", "BOLTZ_XL_TRANS_ROWS": "256", "BOLTZ_XL_COND_ROWS": "256",
           "BOLTZ_XL_COND_FP32": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
# big = fast minus ONLY the levers with a measured resident-memory cost, plus the memory levers (the tier rule). OUT by
# measurement: the sampler group — the roll-out's graph pool and the DiT hoist's level-2 cache are resident memory (dit_hoist +4.7 GB at 1200
# tokens; the fused DiT step and the in-graph Kabsch ride the roll-out: NEEDS). OUT pending their composition with the XL levers (the row-chunked pair
# Transition / diffusion conditioner act on the STOCK modules the fused pair track and the conditioner fold replace): pairblock / fused_transition /
# pairfuse / condproj — big's attention lever stays the flash patch. Everything else of fast rides: the trunk levers, the fused TriMul and MSA-module
# kernels, the layer_norm replica, the trunk CUDA graphs (<= 300 tokens, idle above by name), the WASTE hoists (the constructor skip is the
# process-start win; the cast / mask hoists step aside by name where the memory line owns the module), the fused dim-64 MSA transition, the
# persistent featurizer last, + TF32 products for the eager score model's fp32 GEMMs (dit_tf32: -17 % fwd at 1200 tokens on the eager sampler, no memory;
# below the 2 % rule on fast's fused bf16 sampler, so fast does not carry it) and fast's fused atom encoder/decoder kernels with their buffers released per
# sample() (peak-neutral on sm_90; off by name on sm_80, where they hold device memory outside the allocator: CARD_DROPS["8.0"]["big"]); the fused PWA
# is tabled and off by name (MODES["big"]["off"]: a measured allocator cost).
BIG_ATTN_BASES = ("flash", "pairtrack")                    # the memory line's pair-stack base, by name: `flash` = the staged flash triangle-attention patch inside the STOCK Pairformer
                                                             # statements (BOLTZ_TRIATTN=flash; z fp32-resident under boltz's autocast, the row-chunked xl_trans over the stock Transition);
                                                             # `pairtrack` = fast's fused pair track (BOLTZ_PAIRBLOCK=default + BOLTZ_TRANSITION=fast + BOLTZ_PAIRFUSE=bf16: the layer driver on one
                                                             # resident bf16 z per stack call, K2B >= 300 tokens; xl_trans then serves only the stacks the driver hands back, by name)
BIG_ATTN_BASE = "pairtrack"                                    # the shipped base — chosen by measurement (PEAK alloc at the largest measured token count decides)
_PAIRTRACK_WORDS = ("BOLTZ_PAIRBLOCK", "BOLTZ_TRANSITION", "BOLTZ_PAIRFUSE", "BOLTZ_CONF")   # fast's fused pair track + conditioner fold words
_BIG_OUT_WORDS = (_PAIRTRACK_WORDS if BIG_ATTN_BASE == "flash" else ())   # the pair track on the flash base (the sampler group rides up to the card's token ceiling below)
SAMPLER_MAX_TOKENS_ENV = "BOLTZ_SAMPLER_MAX_TOKENS"                      # the sampler group's token ceiling on the memory row (bz_sampler / boltz_dit_hoist read it): predictions above it sample on
                                                                         # the stock eager loop with the fused step, BY NAME; the row carries `card` and _tabled states the
                                                                         # number for the probed card (BIG_SAMPLER_CEILINGS); rollout's own placement word (registry `words`: leaves with it)
BIG_SAMPLER_GROUP = ("rollout", "dit_hoist", "align_jacobi64")        # fast's sampler levers with a measured resident-memory cost (the roll-out's graph pool + pre-drawn tables, the hoist's cache:
                                                                         # +1.16 / +2.69 / +0.35 GiB alloc, +3.5-4.4 GB NVML at 400 / 800 / 1200 tokens) — admitted below the ceiling: where fast's
                                                                         # whole pass fits the card the memory row IS fast's composition; above it they step aside
BIG_SAMPLER_CEILINGS = ((79_000, 2000), (39_000, 1200))              # (card memory floor, MiB as nvidia-smi reports the total; max tokens), BY NUMBER: the largest measured token count at which
                                                                         # fast's measured pass peak (NVML high-water, the binding figure) fits the card under the kit's headroom (boltz_graph_patch
                                                                         # HEADROOM_MARGIN_FRAC: 12 % of the card kept free). fast's NVML peak at 400 / 800 / 1200 / 1400 / 2000 tokens: H100 80GB
                                                                         # 12019 / 22887 / 30279 / 34743 / 50465 MiB, A100-80GB 15981 / 28875 / 33545 / 37993 / 53601 MiB; at 3000 both exceed the
                                                                         # card (78947 / 80915 MiB, then out of memory) -> 80 GB cards (81559 / 81920 MiB total, 71772 / 72089
                                                                         # usable): 2000; 40 GB cards (40960 MiB, 36045 usable; the A100's own peaks bind: 33545 fits, 37993 does not): 1200;
                                                                         # smaller cards: the group leaves the row by name (card_memory_below:39000MiB). Between two measured token counts: the lower.
_BIG_DIT_WORD = _FAST_ROW["BOLTZ_SAMPLER_DIT"] + ",weights=sample"      # fast's fused bf16 token-transformer step served from the STOCK eager loop (per-sample() pair-bias pack; the roll-out is
                                                                         # not its host), its packed weights built at sample() entry and dropped at its exit: nothing of it resident at the item
                                                                         # peak (the trunk's): PEAK alloc / NVML unchanged
_BIG_PAIRFUSE_WORD = "bf16,trimul=core.big,triatt=core.big,transition=core.big"  # (+ the tri-attention and transition sites by the providers' `big` TIER words) the PAIRFUSE driver's word on this row: fast's pair-track word with the TriMul site by the core provider's
                                                                         # `big` TIER word — the provider's memory-tier pick for this row (row v4 at c_z 128 on 9.0 / 8.0,
                                                                         # where its `fast` word serves `native` on 9.0); residency and the tri-attention pick as fast's
_BIG_ATOM_WORDS = {"BOLTZ_ATOM_GEMM": (_FAST_ROW["BOLTZ_ATOM_GEMM"], ("BOLTZ_ATOM_RELEASE", "sample"))}   # fast's fused atom kernels' placement word on this row: static buffers released at every sample()
                                                                         # return (boltz_atom.set_release) — nothing of them resident under the next prediction's trunk peak: PEAK alloc +0.01 GiB,
                                                                         # NVML +8 MiB at 1200 / 2000 tokens on sm_90 for -1.0 / -1.1 s of sampler. On sm_80 the kernels hold +3.5 GB
                                                                         # of device memory outside torch's allocator: there the lever leaves the row by name (CARD_DROPS["8.0"]["big"]) and these
                                                                         # words with it (registry `words`)
_BIG_ROW = {**{kk: vv for k, v in _FAST_ROW.items() if k not in _BIG_OUT_WORDS and k not in ("BOLTZ_WRITER", "BOLTZ_PREFETCH")
                 for kk, vv in ([(k, _BIG_PAIRFUSE_WORD)] if k == "BOLTZ_PAIRFUSE" else [(k, "big")] if k == "BOLTZ_FPF_TRIMUL_PROVIDER" else [(k, _BIG_DIT_WORD)] if k == "BOLTZ_SAMPLER_DIT" else [(k, v), _BIG_ATOM_WORDS[k][1]] if k in _BIG_ATOM_WORDS
                                else [(k, v), (SAMPLER_MAX_TOKENS_ENV, "card")] if k == "BOLTZ_SAMPLER_ALIGN" else [(k, v)])},
              **(_F2_ROW if BIG_ATTN_BASE == "flash" else {}), "BOLTZ_PRECISION": "dit_tf32", **_XL_ROW,
              "BOLTZ_WRITER": _FAST_ROW["BOLTZ_WRITER"], "BOLTZ_PREFETCH": _FAST_ROW["BOLTZ_PREFETCH"]}   # fast's words in fast's order, the flash patch, the memory line, the featurizer last (at n_gpu > 1 the row statements replace the whole-tensor kernels: rowpair.REPLACED_LEVERS; TP_DROPS names the rest)
_BIG_ROW["BOLTZ_TRANSITION"] = "big"                                        # the transition adapter binds the core provider by the row's own TIER word (fast: `fast`); under the PAIRFUSE driver the C=128 sites are the driver's (transition=core.big above), the adapter keeps the MSA rows of width 64 and the conditioner's calls


# The staging set per route: which carried kit files the worker process needs in its working directory. Paths are relative to boltz2/opt/
# (registry.TRUNK / DIT). The engine adapters (boltz2_opt.<name>, worker_launch --attach) import their carried files from opt/forward/<add-on>/
# themselves (stack.kit_path): nothing of theirs is staged.
_WORKER_FILES_COMMON = [
    f"{_TRUNK}/boltz_trunk_levers.py",
    f"{_DIT}/src/boltz_graph_patch.py", f"{_DIT}/src/boltz_dit_hoist.py", f"{_DIT}/src/make_worker_variant.py",
]
_F2_FILES = [f"{_TRUNK}/boltz_flash_triattn_patch.py"]
# the memory levers are installed by the engine adapter boltz2_opt/big.py on the core's memory primitives (attached by boltz2_opt.worker_launch
# --attach xl right after the worker's boltz import — no worker bytes edited, no kit file staged); until its levers are in the attach refuses by name
# Kernels a staged file imports (boltz_flash_triattn_patch.py: `import flash_triattn`) are the core's carried copy, routed BY NAME
# (opt_core.kernels; this tree carries no copy of a kernel the core ships): stack.stage holds the route before the launch,
# boltz2_opt.worker_launch installs it in the worker before the first import.
ROUTED_KERNELS: Dict[str, List[str]] = {"exact": ["fpf_trimul"], "fast": [], "big": ["flash_triattn"]}   # the TriMul kernels of every row are the core provider's, reached by their core names (opt_core.kernels.trimul; no route, no export of this tree's); fast's attention core (fpf_triatt_k2b) and the block's prologue/epilogue/transition cells are routed by opt_core.attn.pair_fused itself from the core copy; the PAIRFUSE driver's TriMul provider is the routed fpf_trimul_v4
# A routed kernel's required exports (opt_core.kernels.exports: the sums file names the parameter): this tree's per-kit data for it, opt/-relative.
KERNEL_EXPORTS: Dict[str, Dict[str, str]] = {}                    # this tree exports no per-kit data for a routed kernel (no cell table of its own: the core provider's tables serve every part)

# The attachments (worker_launch.ATTACH) per row, in activation order. Hooks on one trigger module run in this list's order: trimul
# (boltz.model.layers.triangular_mult) / transition / pairblock (their layer modules) fire first, pairfuse at boltz.model.modules.trunkv2, sampler and
# ditexact at boltz.model.modules.diffusionv2, the rest right after the worker's own model import (boltz.model.models.boltz2) — all before the worker
# applies boltz_trunk_levers (bz_worker_lev.py:63), whose class patches then land over them. Within the model-import group: the template guard first;
# waste BEFORE msa (the fused MSA-module kernels' by-name stock fallbacks — pwa above its int32 limit — land on the hoisted stock bodies, still bitwise)
# and BEFORE msa2 (trans2 / trans2x take the dim-64 Transitions from whatever forward is installed and hand every other call back to it); graph after
# the layer adapters so a captured stack re-issues their kernels; atom last (instance-level on the structure module, armed before the model is built).
_EXACT_ATTACH = ["templ", "trimul", "transition", "pairblock", "triattn_exact", "sampler", "ditexact", "exactln", "waste", "msa2", "graph", "templskip", "atom", "writer", "prefetch"]
_FAST_ATTACH = ["templ", "trimul", "transition", "pairblock", "pairfuse", "sampler", "exactln", "waste", "msa", "msa2", "conf", "graph", "templskip", "atom", "writer", "prefetch"]

MODES: Dict[str, dict] = {
    "off": {
        "tier": None,
        "promise": "stock — no environment set, no lever applied; the upstream CLI in a clean, proven subprocess (stock_pred.py)",
        "levers": [],
        "env": {},
        "route": "stock_cli",
    },
    "exact": {
        "tier": 1,
        "promise": ("byte-identical to the stock as shipped (upstream's fused cuEquivariance kernels on, production defaults, no recipe): mmCIF "
                    "bytes, PAE and pLDDT arrays, confidence fields, wherever that stock reproduces itself run-to-run (it does at b200-b1400 on "
                    "H100); where it does not, within its own seed-to-seed spread. The pair track runs the core's exact-class cells (the TriMul by the core "
                    "provider's exact tier word, the fused transition, the fused triangle-attention prologue/epilogue around cuEquivariance's own attention core, the "
                    "ATen layer_norm replica), the MSA module's chunked paths their hoisted stock bodies and the exact fused dim-64 transition, the trunk "
                    "stacks replay as CUDA graphs up to 300 tokens, the sampler is the boundary-graph roll-out with the DiT hoist, the token transformer's "
                    "parallel-branch schedule and the atom-attention gather / glue hoists"),
        # the activation order on the worker route: the attachments (below), then the trunk levers at worker import (bz_worker_lev.py:63), then the
        # sampler block the variant script inserts (make_worker_variant.py:6-9,20-26: the graph patch inert — BOLTZ_GRAPH_DIFFUSION unset — the hoist
        # outermost over the roll-out's AtomDiffusion.sample)
        "levers": ["resid", "mask2", "rollout", "dit_hoist", "align_aligncap", "dit_par", "dit_mask", "dit_sba", "dit_smx", "dit_glue", "fpf_trimul_exact", "fused_transition", "pairblock", "triattn_exact", "templ_skip", "exactln", "exactln_resid",
                   "graph_trunk", "waste_chunkcast", "waste_opmmask", "waste_opmdiv", "waste_ctorskip", "msa_pwa_exact", "msa_trans2_exact", "atom_keys_gather", "atom_glue_hoist", "writer_overlap", "prefetch"],
        "env": dict(_EXACT_ROW),
        "off": {},                                   # row-level ablation entries {lever: reason} (see resolve); empty = the row as composed
        "route": PINNED_ROUTE,
        # the DEFINITIONS' base; bz_worker_levf2.py (fast / big) differs from it by one line (levf2:64, the flash patch import, which
        # applies nothing unless BOLTZ_TRIATTN=flash).
        "worker_base": f"{_TRUNK}/src/bz_worker_lev.py",
        "variant": "bz2dit",
        "kernels": "on",
        "stage": list(_WORKER_FILES_COMMON),
        "attach": list(_EXACT_ATTACH),
    },
    "fast": {
        "tier": 2,
        "promise": ("exact's composition with the Tier-2 cells: every C=128 pair stack (trunk, MSA-module pair layers, confidence) driven on ONE resident bf16 "
                    "pair tensor per stack call (the PAIRFUSE layer driver over the core's K2B flash triangle attention at and above 300 tokens — cuEquivariance's "
                    "below — the TriMul by the core provider's fast tier word and the fused transition), the templates' C=64 stack and anything the driver "
                    "hands back on the class-level block adapters; the MSA module's outer-product-mean and pair-weighted averaging through the carried bundle's "
                    "fused Triton kernels, its dim-64 transitions through the fused kernel; the diffusion token transformer as the fused bf16 step inside the "
                    "roll-out, the atom encoder/decoder as fused Triton kernels at bf16 operands; numerics within the engine's Tier-2 band vs the shipped stock "
                    "(judged against stock's own seed-to-seed band per target); below the gates the pair track is identical to `exact`"),
        "levers": ["resid", "mask2", "rollout", "dit_hoist", "align_jacobi64", "dit_fused", "pairblock", "flash_triattn", "pairblock_c64", "templ_skip", "fused_transition", "fpf_trimul", "fpf_opm", "fpf_pwa", "pairfuse",
                   "exactln", "graph_trunk", "waste_chunkcast", "waste_opmmask", "waste_opmdiv", "waste_ctorskip", "msa_trans2", "condproj", "atom_keys_gather", "atom_fused", "atom_gemm", "writer_overlap", "prefetch"],
        "env": dict(_FAST_ROW),
        "off": {},
        "attach": list(_FAST_ATTACH),
        "route": PINNED_ROUTE,
        "worker_base": f"{_TRUNK}/src/bz_worker_levf2.py",
        "variant": "bz2dit",
        "kernels": "on",
        "stage": list(_WORKER_FILES_COMMON) + list(_F2_FILES),
    },
    "big": {
        "tier": 2,
        "promise": ("the memory mode, composed on fast (Tier 2): fast's levers — the roll-out, the DiT hoist and their rider (measured resident memory) up to the card's "
                    "token ceiling only (BIG_SAMPLER_CEILINGS: where fast's whole pass fits the card, by number; above it the stock eager loop with the fused step), the trunk levers, fast's fused pair track (the fused triangle-attention block with the K2B flash core from 300 tokens, the fused "
                    "transition, every C=128 pair stack driven on one resident bf16 z: PAIRFUSE), the fused TriMul, the fused outer-product mean and dim-64 MSA "
                    "transition, the layer_norm replica, the WASTE constructor skip, the trunk CUDA graphs (<= 300 tokens), the persistent featurizer, on the stock eager "
                    "sampler with fast's fused bf16 token-transformer step (dit_fused, its packed weights built and dropped per sample(): weights=sample) and TF32 products elsewhere "
                    "(dit_tf32) and fast's fused atom encoder/decoder kernels, their buffers released per sample() (peak-neutral on sm_90; off by name on sm_80: measured device memory outside the allocator); "
                    "the fused PWA and the conditioner fold off by name (a measured allocator cost / the row-chunked conditioner's site) — + the memory levers (big.BIG_LEVERS: "
                    "the row-chunked Transition serves the stacks the driver hands back); peak GPU memory at or below the stock's, or an input size ceiling above the "
                    "stock's; slower than fast; the numerics are Tier 2: within the stock's own seed-to-seed band"),
        # the levf2 base's order: trunk levers (levf2:63) -> [graph patch, hoist: inert, BOLTZ_GRAPH_DIFFUSION / BOLTZ_DIT_HOIST unset,
        # make_worker_variant.py:21,25] -> flash patch (levf2:64); the engine adapters attached at the boltz import in the row's `attach` order (worker_launch
        # --attach): the memory levers after the WASTE Transition patch (xl_trans row-chunks whichever Transition.forward it finds), the featurizer last.
        # The allocator setting is the process's, read at CUDA init.
        "levers": (["resid", "mask2", "rollout", "dit_hoist", "align_jacobi64", "dit_fused", "templ_skip", "flash_triattn", "fpf_trimul", "fpf_opm", "fpf_pwa", "exactln", "graph_trunk", "waste_chunkcast", "waste_opmmask", "waste_opmdiv", "waste_ctorskip",
                    "msa_trans2", "atom_keys_gather", "atom_fused", "atom_gemm", "dit_tf32", "xl_trans", "xl_cond", "xl_free", "relpos_lazy", "expandable_segments", "writer_overlap", "prefetch"]
                   if BIG_ATTN_BASE == "flash" else
                   ["resid", "mask2", "rollout", "dit_hoist", "align_jacobi64", "dit_fused", "pairblock", "flash_triattn", "pairblock_c64", "templ_skip", "fused_transition", "fpf_trimul", "fpf_opm", "fpf_pwa", "pairfuse", "exactln", "graph_trunk", "waste_chunkcast", "waste_opmmask",
                    "waste_opmdiv", "waste_ctorskip", "msa_trans2", "condproj", "atom_keys_gather", "atom_fused", "atom_gemm", "dit_tf32", "xl_trans", "xl_cond", "xl_free", "relpos_lazy",
                    "expandable_segments", "writer_overlap", "prefetch"]),      # fast's order either way: the fused DiT step served from the stock eager loop (dit_fused, weights=sample); on the pair-track base fast's fused
                                                              # block / transition / layer driver ride and condproj is tabled (it leaves by name: xl_cond takes its site)
        "env": dict(_BIG_ROW),
        "sampler_ceiling": True,                                             # the sampler group serves predictions up to BIG_SAMPLER_CEILINGS[card] tokens (_tabled states the number; above it: the eager loop)
        "off": {"rollout": "rule:big_no_graphs",                       # the memory mode's rule (no CUDA graphs in big): the sampler roll-out's whole-loop graph leaves big BY NAME at
                "dit_hoist": "needs:rollout",                            # the hoist's level-2 cache is read inside the roll-out's captured loop; it leaves with the roll-out (as on the xP line)
                                                                             # every size (the DiT hoist follows: it rides the roll-out); the token-ceiling table below stays in place, inert; the fused
                                                                             # bf16 step keeps serving from the stock eager loop (weights=sample)
                "graph_trunk": "rule:big_no_graphs",                   # the same rule: CUDA graphs are allowed in exact / fast, never in the memory mode — the trunk graphs
                                                                  # (<= 300 tokens; idle above by name) leave big BY NAME with their words and the `graph` attachment; bytes unaffected
                                                                  # (exact-class lever), their capture pool (0.14 GiB at 199 tokens) no longer resident in the memory mode
                "fpf_pwa": "measured_memory_2.8GiB_alloc_peak",            # the fused pair-weighted averaging is the memory row's allocator PEAK: +2.77 GiB alloc / +4.3 GiB reserved / +4.4 GB NVML at
                                                                             # 1200 tokens for -0.76 s/item (the stock head-chunked PWA runs instead; fpf_opm stays: -1.0 s/item, alloc-neutral) — off BY NAME
                # atom_fused / atom_gemm RIDE this row (BOLTZ_ATOM_RELEASE=sample, tabled above: peak-neutral on sm_90) and leave BY NAME on sm_80 (CARD_DROPS["8.0"]["big"]:
                # +3.5 GB of device memory outside the allocator there); on the composed row the fused kernels serve every atom layer and the DiT hoist is off by rule, so
                # atom_keys_gather has no caller and leaves by name (CONDITIONAL_RIDERS) — where the fused kernels are off by card the stock layers call to_keys and it rides
                },
        "route": PINNED_ROUTE,
        "worker_base": f"{_TRUNK}/src/bz_worker_levf2.py",
        "variant": "bz2dit",
        "kernels": "on",
        "attach": (["templ", "trimul", "sampler", "exactln", "waste", "msa", "msa2", "graph", "templskip", "atom", "precision", "xl", "writer", "prefetch"] if BIG_ATTN_BASE == "flash" else
                   ["templ", "trimul", "transition", "pairblock", "pairfuse", "sampler", "exactln", "waste", "msa", "msa2", "conf", "graph", "templskip", "atom", "precision", "xl", "writer", "prefetch"]),   # fast's order for fast's adapters; the precision unit (site __call__, order-free); the memory levers after every Transition patch they wrap (xl_trans calls the forward it finds installed per row block); the featurizer last
        "stage": list(_WORKER_FILES_COMMON) + list(_F2_FILES),
    },
}

# The kit worker's arguments of the kits' probe arm: bz_worker.py --mode fast --kernels off --num_workers 1 --pipeline 1 --keep_on_gpu 1 (bz_worker_lev.py:29-34).
# The kernel state of a launch is the mode row's (`kernels`; worker_args()): `on` for exact / fast / big (the probe arm's own `off` is the kits' kernels-off line, no mode's).
WORKER_ARGS = {"mode": "fast", "kernels": "off", "num_workers": "1", "pipeline": "1", "keep_on_gpu": "1"}

# Evidence switches, per mode — not levers: they change what a kit prints, never what it computes. The flash patch prints its own per-call
# counters (flash_calls / stock_calls with the stock reason per call, exceptions) as one JSON line at interpreter exit only under
# BOLTZ_TRIATTN_REPORT=1 (boltz_flash_triattn_patch.py:33,267-265); stack.evidence reads that line for `fast`.
EVIDENCE_ENV = {"off": {}, "exact": {}, "fast": {"BOLTZ_TRIATTN_REPORT": "1"}, "big": {"BOLTZ_TRIATTN_REPORT": "1"}}
# Verbose words of the same family — they change what a kit file prints on stderr, never what it computes; no row sets them and a caller's copy
# is stripped with the lever words (stock/PINS.json must_be_absent_prefixes): the per-step graph patch's BOLTZ_GRAPH_DEBUG, the hoist's
# BOLTZ_DIT_HOIST_DEBUG (opt/forward/dit_hoist), the roll-out's BOLTZ_SAMPLER_DEBUG (opt/forward/sampler), the WASTE module's BOLTZ_WASTE_VERBOSE.
VERBOSE_WORDS = ("BOLTZ_GRAPH_DEBUG", "BOLTZ_DIT_HOIST_DEBUG", "BOLTZ_SAMPLER_DEBUG", "BOLTZ_WASTE_VERBOSE")

# The ×P line's placement words (`--mode big --n_gpu P > 1`): the core's host-parking levers of the row-sharded pair stack, read by
# ``opt_core.mem.rowpair`` under these names (opt_core/mem/rowpair/API.md "Levers read from the environment"; this package maps nothing —
# the core's own words are the kit surface). ``stack.child_env`` sets each at ``n_gpu > 1``; they are the row's alone (a caller's copy is
# stripped with the other lever words, stock/PINS.json must_be_absent_prefixes); ``rowpair.report()["tp_exports"]`` records the words in force
# in each rank and the core's schedule census prints what they did (``park_z_init=``, ``msa_host=``, ``conf_ztrunk=``). Nothing is set at ``n_gpu = 1`` (nothing installed).
TP_EXPORTS = {
    "ROWPAIR_TRIATT_STAGE": "once", # the tri-attention kernel's bias operand staged ONCE per gathered plane per orientation (core triatt_update_ arms the StageSlot; per_call = staged at every call); census triatt_stage=once:<kernel> triatt_stage_planes=
    "ROWPAIR_HOST_SLAB": "lease",   # the u-sized pinned host copies (z_init/z_trunk/MSA-entry parks, the TriMul row mirror) share ONE leased slab per rank
    "ROWPAIR_RANK_THREADS": "auto", # per-rank CPU threads = cores / ranks (the rank entry applies it; census rank_threads=)
    "ROWPAIR_PARK_ZINIT": "recompute",   # no z_init tensor anywhere — a recycle's row block re-runs the init statement (bitwise the parked bytes); census park_z_init=recompute. The parked form: the initial pair shard z_init [R, N, 128] fp32 PARKED on pinned host between recycles (core trunk.ShardPark inside
                                    # trunk.run_trunk_sharded, which rowpair._trunk_sharded drives): the device holds z + the row blocks the recycle
                                    # statement reads, not z + z_init — −512·N²/P bytes per rank through the template / MSA / Pairformer stages
                                    # (1.98 GiB at 4,076 tokens P = 4; 8.4 GiB at 11,856 tokens P = 8); census `park_z_init=host_pinned|host_pageable:<kind>|host|device` (the core's ShardPark.where; `host` = a CPU shard, `device` = not parked)
    "ROWPAIR_MSA_HOST": "rank0",    # the raw MSA features (msa, has_deletion, deletion_value, msa_paired: [1, S, N], ≈25·S·N bytes with msa_mask) HOST-RESIDENT: the batch's
                                    # H2D keeps them on rank 0's pinned host (ranks > 0 hold zero-row placeholders) and each cycle only the rows the MSA module
                                    # uses reach the device — selected on the host when the module subsamples (num_subsampled_msa of S), all S rows when it
                                    # does not (the worker line: subsample off) — rank 0 stages them, the others receive them by chunked broadcast (core
                                    # msa_host; adapter rowpair_msa); census `msa_host= msa_rows=<n>/<S> msa_host_cycle_gib= msa_host_rows=`; `=all`: every
                                    # rank keeps its own host copy; `=0`: the stock device placement
    "ROWPAIR_CONF_PARK_ZTRUNK": "1",  # the trunk pair shard z [R, N, 128] fp32 PARKED on pinned host at the roll-out entry (its device storage released before the conditioned
                                    # pair rows z_cond are allocated) and served per row block to the diffusion conditioning and to every confidence pass's
                                    # embedding; dropped after the last pass (core heads.ZTrunkPlan, bound by rowpair._ztrunk_plan / rowpair_heads): the roll-out
                                    # holds z_cond + the distogram rows, a confidence pass ONE pair shard (its own pair input) at any diffusion_samples —
                                    # −512·N²/P bytes per rank from the roll-out entry to the end of the forward (1.98 GiB at 4,076 tokens P = 4; 8.4 GiB at
                                    # 11,856 P = 8); census `conf_ztrunk_entry=parked:host_pinned conf_ztrunk=parked:host_pinned[,…] conf_ztrunk_host_gib=`
    "ROWPAIR_TRANSPOSE_INPLACE": "1", # the pair block's ending-node transposes (z -> z^T for the ending triangle attention, and back) as pairwise block swaps into the shard's OWN storage
                                    # (core ring.transpose_shard_inplace_, chosen inside pairstack.pair_block_ — the one pair-block statement rowpair._pair_layer drives for the
                                    # template, MSA, trunk and confidence pair stacks): no second shard around the ending attention (−512·N²/P bytes of transient per
                                    # pair block: −1 u) under a different comm schedule (pairwise XOR block swaps, next_pow2(P)−1 rounds; its wall effect is
                                    # not timed here); census `pairstack_transpose=inplace` (`=p2p`: the all-to-all form, `=0`)
    "ROWPAIR_FREE_ZTRUNK": "1",       # when no park is live (CONF_PARK=0 or a shard that cannot be parked): the LAST confidence pass's pair input OVERWRITES the trunk
                                    # shard's storage in place (zero copies, zero host bytes; census `conf_ztrunk=inplace`) — one sample is embedded per pass, so at
                                    # diffusion_samples S > 1 the passes before the last read the resident shard and say so (`resident:free_declined:not_last,…,inplace`).
                                    # Either lever consumes the trunk rows: dict_out["z"] becomes rowpair.ConsumedRows (refused by name if read; `=0 =0` keeps it)
    "ROWPAIR_MSA_M_LAYOUT": "token_sharded",  # the MSA representation m TOKEN-SHARDED across the ranks (adapter rowpair._msa_rows / rowpair_msa.msa_input on core msa.pwa_rows /
                                    # opm_rows_budgeted(a_local) / msa_transition_rows): rank q one-hots, embeds and updates only its token columns m[:, :, r0_q:r1_q, :]
                                    # (the pair shard's rows); the pair-weighted averaging gathers one 128-row S-chunk of m at a time (`ROWPAIR_PWA_S_CHUNK`), the
                                    # outer-product mean gathers its b operand [S, N, 32] once — with the MSA subsample off (the worker line) S = every MSA row (up to
                                    # 8192): −1024·S·N·(1−1/P) bytes per rank in the MSA module (45 GiB -> 45/P GiB + one gathered bf16 OPM operand 64·S·N B = 2.8 GiB and one 128-row PWA chunk ≈ 1 GB at S = 8192,
                                    # N = 5780); census `msa_m=token_sharded msa_cols=<r0>:<r1>/<N> msa_m_gib= pwa_s_chunk= opm_a=local opm_b_gather_gib=`;
                                    # `=replicated`: m [1, S, N, 64] whole on every rank (the m-update's token rows all-gathered instead); no `=0` form (a layout, not a park)
    "ROWPAIR_TRIATT_CORE": "tier:big",      # the core adapter's TIER DOOR: kernels.triattn's row for the word `big` per row window (9.0: the core's CUDA triangle-attention row by the cell table;
                                    # a refusal / an unmeasured card steps aside BY NAME to the flash path: schedule word triatt_core_tier_off). At the ×P row window
                                    # ([1,128,4,N,32] bf16 + whole-plane bias) the table names the CUDA row at every token count it holds; above its largest cell the
                                    # nearest cell serves with an UNCOVERED_CELL alert. The other words:
                                    # "flash_triattn" = the triangle-attention core of every row block: the fused flash kernel through the core's adapter (opt_core.mem.rowpair.triatt.
                                    # attention_core; Tier 2 vs the eager core, run-to-run repeatable; a call the kernel does not serve runs the eager statement, named on
                                    # the core's `LEVER name=F1.flash_triattn …` line) | cueq | torch (= the eager statement for every call: the ×P line before this word)
    "ROWPAIR_TRIMUL_KERNELS": "fpf_v4",         # the triangle multiplications of every row block: the fused K1 -> bmm -> K3 rows (opt_core.mem.rowpair.trimul_fused over the core's
                                    # own fpf_trimul_v4 cell table; Tier 2; pairs under the core's size gate (2048 tokens) and any unit the provider declines run the torch
                                    # statements, named on the core's `LEVER name=F2.trimul_rows …` line) | torch (= the torch statements: the ×P line before this word)
}


def placement_findings(exports: Dict[str, Optional[str]], schedule: Dict[str, object]) -> Tuple[List[str], List[str]]:
    """READ the ×P placement words off a rank's schedule census against the words in force in that rank (``tp_report.tp_exports``: the
    ``TP_EXPORTS`` names as the rank process carried them, caller overrides included) -> ``(fallbacks, problems)``. A FALLBACK is a word that
    was in force and acted in a degraded form the core NAMED (a page-lock refused -> ``host_pageable:<kind>``; a shard that could not be parked
    -> ``resident:<why>``; the park lever set but the initial shard left on the device): the run is a partial activation (``stack.
    partial_activation``: exit 3 unless ``--allow-partial``). A PROBLEM is a word in force whose census is absent or contradicts it (the
    binding did not engage: ``msa_host=off`` under ``ROWPAIR_MSA_HOST=rank0``; ``msa_m=replicated`` under ``ROWPAIR_MSA_M_LAYOUT=token_sharded``;
    a confidence pair stack that ran on a working COPY of the rows).
    A word the caller set to ``0`` expects nothing. One reader for the evidence (``stack.evidence``) and the tests."""
    fallbacks: List[str] = []
    problems: List[str] = []
    on = lambda k: str(exports.get(k) or "").strip() not in ("", "0")                 # noqa: E731
    if on("ROWPAIR_PARK_ZINIT"):
        w = schedule.get("park_z_init")
        if w is None or str(w) == "device":
            fallbacks.append(f"park_z_init={w} under ROWPAIR_PARK_ZINIT=1 (the initial pair shard stayed on the device)")
        elif str(w).startswith("host_pageable"):
            fallbacks.append(f"park_z_init={w} (pinned host refused: a pageable park, named by the core)")
    want = str(exports.get("ROWPAIR_MSA_HOST") or "").strip()
    if want not in ("", "0"):
        want = "all" if want == "1" else want
        got = schedule.get("msa_host")
        if str(got) != want:
            problems.append(f"msa_host={got} under ROWPAIR_MSA_HOST={want} (the raw-MSA host placement did not engage at the MSA module)")
        where = str(schedule.get("msa_host_where") or "")
        if "host_pageable" in where:
            fallbacks.append(f"msa_host_where={where} (pinned host refused for a raw MSA feature: pageable, named by the core)")
    words = [w for w in str(schedule.get("conf_ztrunk") or "").split(",") if w and w != "-"]
    entry = schedule.get("conf_ztrunk_entry")
    if on("ROWPAIR_CONF_PARK_ZTRUNK"):
        if entry is None or not str(entry).startswith("parked:"):
            fallbacks.append(f"conf_ztrunk_entry={entry} under ROWPAIR_CONF_PARK_ZTRUNK=1 (the trunk shard was not parked at the roll-out entry)")
        elif "host_pageable" in str(entry):
            fallbacks.append(f"conf_ztrunk_entry={entry} (pinned host refused: a pageable park, named by the core)")
        bad = [w for w in words if not w.startswith("parked:")]
        if bad:
            fallbacks.append(f"conf_ztrunk={','.join(words)} under ROWPAIR_CONF_PARK_ZTRUNK=1 (a confidence pass read a resident trunk shard)")
    elif on("ROWPAIR_FREE_ZTRUNK") and words and words[-1] != "inplace":
        fallbacks.append(f"conf_ztrunk={','.join(words)} under ROWPAIR_FREE_ZTRUNK=1 without a park (the last confidence pass did not embed in place)")
    want_m = str(exports.get("ROWPAIR_MSA_M_LAYOUT") or "").strip()
    if want_m and want_m not in ("token_sharded", "replicated"):          # rowpair_msa.m_layout refuses the same word by name at install
        problems.append(f"ROWPAIR_MSA_M_LAYOUT={want_m}: not a layout (token_sharded | replicated)")
    elif want_m:
        got = schedule.get("msa_m")
        if str(got) != want_m:                                              # absent = the MSA module never recorded its layout under the word: a problem too
            problems.append(f"msa_m={got} under ROWPAIR_MSA_M_LAYOUT={want_m} (the MSA representation's layout did not engage at the MSA module)")
    if on("ROWPAIR_TRANSPOSE_INPLACE"):
        tf = schedule.get("pairstack_transpose")
        if str(tf) != "inplace":                                                        # absent = no pair block recorded its transpose form: a problem as much as p2p
            problems.append(f"pairstack_transpose={tf} under ROWPAIR_TRANSPOSE_INPLACE=1 (the pair block's ending-node transpose did not run in the shard's own storage)")
    ps = schedule.get("conf_pairstack")
    if ps is not None and str(ps) != "inplace":
        problems.append(f"conf_pairstack={ps} (the confidence Pairformer ran on a working copy of the pass's pair rows: two shard-equivalents stood)")
    return fallbacks, problems


# Why the stock-CLI hook route (one process: `boltz predict` + a sitecustomize hook, or `enable()` before the upstream API) is refused for
# exact / fast / big: the kit fact, named. A hook composing the trunk levers, the sampler levers and the attach-time patches in one stock-CLI
# process would be a deviation from the kits' own line (the persistent worker); it is not shipped (OPEN).
CLI_ROUTE = {
    "off": None,
    "exact": ("this tree carries no stock-CLI hook composing trunk levers + graph sampler + hoist in one process; the kits' line is the "
              "persistent worker, here with its --kernels on argument — use `boltz2-opt pred --mode exact` (route=worker)"),
    "fast": ("this tree carries no stock-CLI hook composing trunk levers + flash tri-attention + graph sampler + hoist in one process; the kits' "
             "line is the persistent worker — use `boltz2-opt pred --mode fast` (route=worker)"),
    "big": ("this tree carries no stock-CLI hook composing the trunk levers, the flash patch and the memory levers' attach-time patches in one process — use "
              "`boltz2-opt pred --mode big` (route=worker)"),
}



# ---------------------------------------------------------------- levers leaving a row BY NAME (per card, per run, per row) ----------------------------------------------------------------
# One mechanism takes a lever off a row, whatever names it: its switch word leaves the row's environment (a unit of a comma-list word when other
# levers of the row still ride the word: registry value's first token; the whole word with its companion words `<SWITCH>_*` otherwise), its
# attachment leaves when no remaining lever rides it (LEVER_ATTACH), the levers that need it leave with it (NEEDS, named `needs:<lever>` / the same
# reason), and the row records it under the key of whoever named it — `card_off` (CARD_DROPS: the probed card), `run_off` (RUN_DROPS: a `boltz
# predict` option of the run), `off` (the row's own ablation entries). The ACTIVE / DRY-RUN lines print one token per key
# (report.off_token: ` off=<lever>:<reason>,…`), the LEVER lines one `state=off reason=<reason>` line per lever (report.lever_lines), the manifest
# records all three. Off restores the exact stock statement (every adapter installs nothing for an absent word).
LEVER_ATTACH = {                                             # lever -> the attachment that installs it (worker_launch.ATTACH); levers of staged kit files (resid, mask2, flash_triattn in big, graph_sampler, dit_hoist) have none
    "fpf_trimul_exact": "trimul", "fpf_trimul": "trimul", "fused_transition": "transition", "pairblock": "pairblock", "pairblock_c64": "pairblock", "triattn_exact": "triattn_exact", "templ_skip": "templskip", "fpf_opm": "msa", "fpf_pwa": "msa",
    "xl_trans": "xl", "xl_cond": "xl", "xl_free": "xl", "relpos_lazy": "xl", "expandable_segments": "xl",
    "writer_overlap": "writer",                                                                 # the background writer: its own attachment (beside the featurizer)
    "prefetch": "prefetch",                                                                     # the persistent featurizer: its own attachment (last in the rows)
    "rollout": "sampler", "align_jacobi64": "sampler", "align_aligncap": "sampler", "dit_fused": "sampler", "condproj": "conf", "tfeat": "conf", "tdummy": "conf", "dit_smx": "ditexact", "dit_glue": "ditexact", "exactln_resid": "exactln", "msa_pwa_exact": "msa2", "dit_par": "ditexact", "dit_mask": "ditexact", "dit_sba": "ditexact",
    "exactln": "exactln", "graph_trunk": "graph", "pairfuse": "pairfuse",
    "waste_chunkcast": "waste", "waste_opmmask": "waste", "waste_opmdiv": "waste", "waste_ctorskip": "waste", "msa_trans2": "msa2", "msa_trans2_exact": "msa2",
    "atom_keys_gather": "atom", "atom_glue_hoist": "atom", "atom_fused": "atom", "atom_gemm": "atom",
    "dit_tf32": "precision", "dit_bf16": "precision", "dit_attn_bf16": "precision", "seq_bf16": "precision",                          # [PRECISION] one row word BOLTZ_PRECISION=<unit,...>
}
CARD_ATTACH_OF = LEVER_ATTACH                                # the per-card drops' name for the same table
NEEDS = {                                                    # lever -> the levers it rides: it leaves a row with any of them, named
    "dit_par": ("dit_hoist",), "dit_mask": ("dit_hoist",), "dit_sba": ("dit_hoist",), "dit_smx": ("dit_hoist",), "dit_glue": ("dit_hoist",),      # the DITEXACT schedule reads the hoist's level-2 cache
    "dit_fused": ("rollout",), "align_jacobi64": ("rollout",), "align_aligncap": ("rollout",),  # the fused DiT step / the Kabsch seams ride the roll-out's captured boundary
    "exactln_resid": ("exactln",), "pairblock_c64": ("pairblock",),                                # the fused residual pass stores through the layer_norm replica
    "triattn_exact": ("pairblock",),                                                          # the exact word serves the fused block's library core: it leaves with the block
    "atom_gemm": ("atom_fused",),                                                            # a dot precision of the fused atom kernels
}
NEEDS_IN = {"fast": {"flash_triattn": ("pairblock",)},      # per mode: fast's flash attention core lives inside the fused block (big: the staged patch on the flash base; on the pair-track base the
                                                             # block hosts it at n_gpu = 1 and the row-sharded trunk's core adapter at n_gpu > 1 (TP_EXPORTS ROWPAIR_TRIATT_CORE) — no rider either way)
            "big": {"dit_fused": ()}}                      # on the memory row the fused DiT step serves from the STOCK eager loop (bz_sampler packs the pair bias at sample() entry before it hands
                                                             # the call to the inner sampler): the roll-out is not its host there — no rider, by name


def _unit(lever: str) -> str:
    """The lever's unit token inside its switch word: the registry value's first word (`keys (a unit of the comma list)` -> keys; `graph` -> graph)."""
    import re as _re
    from .registry import LEVERS as _L
    m = _re.match(r"\s*([A-Za-z0-9_.]+)", str(_L[lever].get("value") or ""))
    return m.group(1) if m else ""


OFF_KEYS = ("card_off", "off", "run_off", "tp_off")            # the by-name maps of a resolved row, in the order the ACTIVE / DRY-RUN tokens print them (report.off_token)


def take_off(row: dict, lever: str, reason: str, key: str, env_names=None, attach=None) -> None:
    """Take `lever` off `row` BY NAME, recorded under row[key][lever] = reason (key: card_off | run_off | off | tp_off). `env_names` / `attach` given
    (CARD_DROPS entries) name the words and the attachment explicitly; otherwise they follow from the registry (switch / value) and LEVER_ATTACH."""
    from .registry import LEVERS as _L
    if lever not in row.get("levers", []):
        return
    row["levers"] = [x for x in row["levers"] if x != lever]
    env = dict(row.get("env") or {})
    sw = _L[lever]["switch"]
    if env_names is not None:
        for k in env_names:
            env.pop(k, None)
    elif sw in env:
        if any(_L[x]["switch"] == sw for x in row["levers"]):            # other levers of the row ride the word: this lever's unit leaves, the word stays
            unit = _unit(lever)
            env[sw] = ",".join(t for t in (w.strip() for w in env[sw].split(",")) if t and t != unit)
        else:                                                              # the word and its companion words (<SWITCH>_MIN_TOKENS, …) leave
            for k in [k for k in env if k == sw or k.startswith(sw + "_")]:
                env.pop(k)
    for k in _L[lever].get("words") or ():                                 # the lever's OWN placement words (registry `words`: atom_fused's BOLTZ_ATOM_RELEASE) leave with it either way
        env.pop(k, None)
    row["env"] = env
    a = attach or LEVER_ATTACH.get(lever)
    if a and a not in {LEVER_ATTACH.get(x) for x in row["levers"]}:
        row["attach"] = [x for x in (row.get("attach") or []) if x != a]
    row.setdefault(key, {})[lever] = str(reason)
    needs = dict(NEEDS); needs.update(NEEDS_IN.get(row.get("mode") or "", {}))
    for dep, on in needs.items():                                          # the levers that ride this one leave with it, named
        if dep in row["levers"] and lever in on:
            take_off(row, dep, (f"needs:{lever}" if key == "off" else reason), key)


# Per compute capability, additive: a lever leaves a mode's row on a card ONLY when it is not bitwise to stock there (exact's contract), BY NAME
# with its reason word (`card_off`). A card absent here runs every row as composed. The card is the probed GPU's, set once per process by stack.gate
# (`set_card`); unset (no GPU probed: the CPU paths) = every row as composed. (On 8.0 exact's fused pair transition and fused triangle-attention block
# serve inside a row range, by name — a per-call size gate inside each lever, transition.CARD_ROWS / pairblock.CARD_ROWS (boltz2_opt.rowfloor), not a
# lever leaving the row; the exact TriMul is the core provider's exact tier word on 8.0 as on 9.0 (its vouched kernel row or the library op by name, per cell); the layer_norm replica proves its bits at its first call on any card, exactln.PROVEN_CC; the token
# transformer's fused softmax / glue replicas and the MSA transition's replica prove theirs per class at run time on 8.0 as on 9.0 (ditexact PROVEN_CC, msa2.EXACT_PIN["cc"]["trans2x"]).
# The MSA PairWeightedAveraging's exact cell is pinned to 9.0 by name, msa2.EXACT_PIN["cc"]["pwa2x"] (its sm_90 candidate set proves one A100 class in three), so on 8.0 that
# unit leaves the exact row; the exact tri-attention lever asks the core provider's `exact` tier word, which names the row per card itself.)
CARD_DROPS: Dict[str, Dict[str, Dict[str, dict]]] = {
    "8.0": {"big": {"atom_fused": {"reason": "measured_memory_nvml_+3.5GB:sm_80"}},                                            # the fused atom encoder/decoder kernels (buffers released per sample()) are peak-neutral on sm_90 (PEAK alloc +0.01 GiB, NVML +8 MiB at 1200 / 2000 tokens for -1.0 / -1.1 s of sampler) but hold +3.5 GB of device memory outside torch's allocator on sm_80 (NVML 13279 -> 16797 MiB at 800 tokens, 19961 -> 23479 at 1200; reserved unchanged; sampler 5.59 -> 3.82 / 6.30 -> 4.29 s): a measured memory cost on THIS card's memory row — the lever leaves by name here with its words (atom_gemm rides it: NEEDS); the stock atom layers and the exact key gather serve
            "exact": {"msa_pwa_exact": {"reason": "pin:cc80"}}},                                                       # the MSA PairWeightedAveraging's exact cell is pinned to 9.0 by name (msa2._pin_word("pwa2x"): bitcmp_failed:400x7311:c / :1200x7410:c on A100, only 800x7382:c proves): its unit leaves the word, trans2x stays. The exact tri-attention lever rides on every card: the core provider's `exact` tier word names the row per card and stack (its bitwise kernel on 9.0 where vouched, the per-head split of the library kernel or the library kernel itself elsewhere — on an unmeasured card the library kernel — by name, never a refusal)
    "10.0": {"exact": {"dit_smx": {"reason": "unproven_cc:sm_100"}, "dit_glue": {"reason": "unproven_cc:sm_100"},                # these replicas are not checked on sm_100: the words leave by name so the row launches; every other
                       "exactln": {"reason": "cc_unproven:sm_100", "env": ("BOLTZ_EXACTLN",), "attach": "exactln"},          # lever of the row runs as on sm_90 (Triton / cuBLAS / CUDA-graph paths with no architecture pin); exactln.PROVEN_CC = (90, 80): its apply refuses by name on any other card — the word leaves instead
                       "msa_pwa_exact": {"reason": "pin:cc100"}, "msa_trans2_exact": {"reason": "pin:cc100"}},
             "fast": {"exactln": {"reason": "cc_unproven:sm_100", "env": ("BOLTZ_EXACTLN",), "attach": "exactln"},
                      "pairblock_c64": {"reason": "no_cell:sm_100", "env": ("BOLTZ_PAIRBLOCK_C64",)}},                                                 # the template (64,4,32) cells exist on 9.0 / 8.0 only
             "big": {"exactln": {"reason": "cc_unproven:sm_100", "env": ("BOLTZ_EXACTLN",), "attach": "exactln"},
                       "pairblock_c64": {"reason": "no_cell:sm_100", "env": ("BOLTZ_PAIRBLOCK_C64",)}}},   # big carries fast's layer_norm replica: the same word leaves on sm_100; no template cells there
}
_CARD: Dict[str, Optional[str]] = {"cc": None, "memory_mib": None}


def set_card(cc: Optional[str], memory_mib=None) -> None:
    """The probed GPU's compute capability ('M.m') and total memory (MiB, nvidia-smi's memory.total; stack.gate sets both once per process);
    None = no card (every row as composed)."""
    _CARD["cc"] = (cc or None)
    try:
        _CARD["memory_mib"] = int(memory_mib) if memory_mib is not None else None
    except (TypeError, ValueError):
        _CARD["memory_mib"] = None


def card_memory_mib():
    return _CARD.get("memory_mib")


def sampler_ceiling(memory_mib) -> Optional[int]:
    """The memory row's sampler-group token ceiling for a card of `memory_mib` total memory (BIG_SAMPLER_CEILINGS: the largest floor at or
    below the card's memory names it); 0 = below every floor (the group leaves the row by name); None = no card probed (as composed)."""
    if memory_mib is None:
        return None
    for floor, tokens in sorted(BIG_SAMPLER_CEILINGS, reverse=True):
        if int(memory_mib) >= floor:
            return int(tokens)
    return 0


def card() -> Optional[str]:
    return _CARD["cc"]


def card_drops(mode: str, cc: Optional[str] = None) -> Dict[str, dict]:
    """``{lever: {reason, env, attach}}`` the row of `mode` does not run on compute capability `cc` (default: the set card); ``{}`` = as composed."""
    return dict((CARD_DROPS.get(cc if cc is not None else (_CARD["cc"] or ""), {}) or {}).get((mode or "").strip().lower(), {}))


# Levers a RUN takes off by name for a `boltz predict` option they cannot carry (settings.worker_settings -> worker.run -> set_run_drops -> resolve()):
# `--use_potentials` steers upstream's diffusion sampler step by step (FK steering / physical guidance, main.py:1310-1311); the roll-out, the
# per-step graph patch and the DiT hoist replay captured work, so on exact / fast that run's sampler is upstream's eager loop — the sampler levers and
# their switches leave the row, and with the hoist the levers that read its cache (NEEDS: dit_par / dit_mask / dit_sba; with the roll-out dit_fused),
# one `LEVER … state=off reason=use_potentials` line each (report.lever_lines); every trunk lever stays on. big's sampler is eager already: nothing
# leaves its row.
def _distinct_sample_chunks(given: dict) -> bool:
    """True when upstream chunks this run's diffusion_samples by max_parallel_samples into chunks of DISTINCT sizes per denoising step (the
    roll-out graph, the graphed sampler and the hoist capture one chunk shape per prediction and step aside by name for such a run:
    settings.upstream_sample_chunks names upstream's rule, settings.SINGLE_SHAPE_LEVERS the levers)."""
    from . import settings as _settings                       # settings imports this module: read it at call time
    d = _settings.stock_defaults()
    try:
        D = int(given.get("diffusion_samples") if given.get("diffusion_samples") is not None else d["diffusion_samples"])
        Mps = int(given.get("max_parallel_samples") if given.get("max_parallel_samples") is not None else d["max_parallel_samples"])
    except (TypeError, ValueError):
        return False
    return len(set(_settings.upstream_sample_chunks(D, Mps))) > 1


RUN_DROPS: Dict[str, dict] = {
    "use_potentials": {"modes": ("exact", "fast", "big"), "levers": ("rollout", "graph_sampler", "dit_hoist")},   # big carries the roll-out and the hoist up to its ceiling: the same words
    "distinct_sample_chunks": {"modes": ("exact", "fast", "big"), "levers": ("rollout", "graph_sampler", "dit_hoist"), "when": _distinct_sample_chunks},
                                                             # a run whose sample chunks have DISTINCT sizes per denoising step (upstream's own chunking of --diffusion_samples by
                                                             # --max_parallel_samples) takes the single-shape sampler levers (settings.SINGLE_SHAPE_LEVERS: the roll-out's whole-loop
                                                             # graph, the graphed sampler, the DiT hoist — one chunk shape per prediction) off BY NAME for that run on every row, their
                                                             # riders with them (NEEDS); the stock eager loop serves every chunk size, as upstream does (exact / fast do not
                                                             # refuse the run). big carries none of them: nothing leaves its row
}
_RUN_OFF: Dict[str, str] = {}                                # lever -> option name, for this process's run (set_run_drops); empty = the row as tabled

# Levers the xP line (`--mode big --n_gpu P`, P > 1: the row-sharded pair stack, rowpair_tp) takes off the row BY NAME, recorded as `tp_off` with the
# reason word (worker.run / cli check state P once: set_n_gpu). At n_gpu > 1 the trunk is the core's row-sharded driver over P rank processes: the
# whole-tensor kernels the row statements replace stay on the row and are named `replaced_by_rowpair` by their own censuses (rowpair.REPLACED_LEVERS:
# the XL levers, fpf_trimul, fpf_opm, fpf_pwa); a lever whose unit has no call site under the sharded trunk, or whose composition with the rank
# processes is not shown, leaves here instead — its word, and its attachment when no other lever rides it — so the xP line launches exactly the
# levers it is shown with (one `LEVER … state=off reason=<word>` line each). P = 1: the row as tabled.
TP_DROPS: Dict[str, Dict[str, str]] = {
    "big": {
        "graph_trunk": "replaced_by_rowpair:the_pairformer_stacks_are_the_sharded_drivers_statements",
        "rollout": "xP_line_unchanged:each_rank_runs_the_stock_eager_sampler",     # the row-sharded line's ranks replicate the stock eager sampler: the sampler group (align_jacobi64 rides the
        "dit_hoist": "xP_line_unchanged:each_rank_runs_the_stock_eager_sampler",   # roll-out: NEEDS) and the fused step are not measured there — they leave by name
        "align_jacobi64": "xP_line_unchanged:each_rank_runs_the_stock_eager_sampler",
        "dit_fused": "xP_line_unchanged:each_rank_runs_the_stock_eager_sampler",
        "atom_fused": "xP_line_not_measured:each_rank_runs_the_stock_atom_layers",  # the fused atom kernels are measured on the single-GPU line only; the xP line keeps the stock atom layers and the exact key
        "atom_gemm": "xP_line_not_measured:each_rank_runs_the_stock_atom_layers",   # gather (unchanged) until measured — both units leave by name (atom_gemm NEEDS atom_fused: named here so the table reads whole)
        "writer_overlap": "xP_line_not_measured:rank_0_writes_inline_as_shipped",   # the background writer is measured on the single-GPU line only; the xP line keeps its inline writer (unchanged) until measured
        "templ_skip": "replaced_by_rowpair:all-dummy_template_passes_elided_in_the_row_statement",   # rowpair builds each rank's template pair rows itself (_template_rows); the class forward this lever binds is never called there
        **({lv: "replaced_by_rowpair:the_pair_stacks_are_the_sharded_drivers_row_statements" for lv in ("pairblock", "pairblock_c64", "fused_transition", "pairfuse")} if BIG_ATTN_BASE == "pairtrack" else {}),
                                                             # on the pair-track base: the fused block / transition / layer driver bind module forwards the row-sharded trunk never calls
                                                             # (its own flash core, streamed TriMul and per-row-block transitions are the row statements) — they leave the xP line by name   # nothing of PairformerModule.forward runs whole at n_gpu > 1 (and a capture would span the ranks' collectives)
    },
}
_TP: Dict[str, int] = {"P": 1}                               # this process's n_gpu (set_n_gpu); 1 = the single-GPU line


def _tabled(m: str) -> dict:
    """MODES[m] as a fresh row (copied containers) with the set card's drops and the row's own ablation entries (R1) taken off by name."""
    row = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in MODES[m].items()}
    row["mode"] = m; row["cli_route"] = CLI_ROUTE[m]
    abl = dict(row.pop("off", {}) or {})
    for lever, d in card_drops(m).items():                    # the set card's drops for this mode, BY NAME (recorded on the row as card_off; never a silent removal)
        take_off(row, lever, d["reason"], "card_off", env_names=tuple(d.get("env") or ()) or None, attach=d.get("attach"))
    if _TP["P"] > 1:
        for lever, why in tp_drops(m).items():                # the xP line's drops for this mode, BY NAME (recorded as tp_off)
            take_off(row, lever, why, "tp_off")
    if row.pop("sampler_ceiling", False) and BIG_SAMPLER_GROUP[0] in row["levers"]:   # the memory row's sampler group up to the card's token ceiling, BY NUMBER (BIG_SAMPLER_CEILINGS)
        nmax = sampler_ceiling(_CARD.get("memory_mib"))
        if nmax is None:
            row["sampler_max_tokens"] = None                  # no card probed (the CPU paths): as composed, the ceiling word unresolved (`card`: the sampler modules read a non-number as no ceiling)
        elif nmax <= 0:
            floor = min(f for f, _ in BIG_SAMPLER_CEILINGS)
            for lever in BIG_SAMPLER_GROUP:                 # a card below every floor: fast's pass does not fit even the smallest ceiling with headroom — the group leaves by name
                take_off(row, lever, f"card_memory_below:{floor}MiB", "card_off")
            row["sampler_max_tokens"] = 0
        else:
            row["env"][SAMPLER_MAX_TOKENS_ENV] = str(nmax); row["sampler_max_tokens"] = nmax
    for lever, why in abl.items():                            # the row's ablation entries {lever: reason}, BY NAME (recorded as off)
        if lever not in MODES[m]["levers"]:
            raise ValueError(f"mode '{m}' off entry names '{lever}', which is not a lever of its row ({','.join(MODES[m]['levers'])})")
        take_off(row, lever, why or "ablation", "off")
    conditional_riders(row, "off")
    return row


# Sampler-side riders NEEDS cannot state (a lever that rides EITHER of two others): the atom key gather serves `to_keys` inside sample() — its callers
# are the DiT hoist's cache build and the STOCK atom encoder/decoder layers. A row that carries the fused atom kernels (they replace the stock layers)
# WITHOUT the hoist leaves it no caller: it leaves BY NAME (its census refuses a unit that serves nothing — the gate stays as it is). On the hoist
# (exact, fast) or on the eager sampler without the fused kernels (big) it serves and stays.
CONDITIONAL_RIDERS = {
    "atom_keys_gather": {"unless_any": ("dit_hoist",), "if_all": ("atom_fused",), "reason": "no_call_site:needs_dit_hoist_or_the_stock_atom_layers"},
    "condproj": {"unless_any": (), "if_all": ("xl_cond",), "reason": "site_taken_by:xl_cond(DiffusionConditioning.forward)"},   # the conditioner fold and the row-chunked conditioner bind the same forward: the memory lever keeps it
}


def conditional_riders(row: dict, key: str) -> None:
    """Take off, BY NAME under row[key], every CONDITIONAL_RIDERS lever whose condition the row (after every other drop) meets."""
    for lever, c in CONDITIONAL_RIDERS.items():
        lv = row.get("levers") or []
        if lever in lv and not any(x in lv for x in c["unless_any"]) and all(x in lv for x in c["if_all"]):
            take_off(row, lever, c["reason"], key)


def run_drops_for(mode: str, given: dict) -> Dict[str, str]:
    """{lever: option} the run of `mode` takes off by name for the `boltz predict` options in `given` (RUN_DROPS, with the levers that ride them:
    NEEDS); {} when none applies."""
    m = (mode or DEFAULT_MODE).strip().lower()
    if m not in MODES:
        return {}
    row = _tabled(m)
    for opt, d in RUN_DROPS.items():
        hit = d["when"](given) if callable(d.get("when")) else given.get(opt)   # an option's presence, or a predicate over the run's options (distinct_sample_chunks)
        if hit and m in d["modes"]:
            for lever in d["levers"]:
                take_off(row, lever, opt, "run_off")
    return dict(row.get("run_off") or {})


def set_run_drops(drops: Optional[Dict[str, str]]) -> None:
    """worker.run states the run's dropped levers once, before anything resolves the row; every resolve() of this process then agrees."""
    _RUN_OFF.clear(); _RUN_OFF.update(drops or {})


def tp_drops(mode: str, n_gpu: Optional[int] = None) -> Dict[str, str]:
    """``{lever: reason}`` the xP line of `mode` takes off by name at ``n_gpu`` > 1 (default: the set n_gpu); ``{}`` at n_gpu = 1 or for a mode TP_DROPS does not name."""
    P = int(_TP["P"] if n_gpu is None else (n_gpu or 1))
    return dict(TP_DROPS.get((mode or "").strip().lower(), {})) if P > 1 else {}


def set_n_gpu(n_gpu) -> None:
    """worker.run / cli check state the run's GPU count once, before anything resolves the row (the launch's rank processes inherit the resolved
    row through their environment and attach list, never this switch); every resolve() of this process then agrees. None / 1 = the single-GPU line."""
    _TP["P"] = max(1, int(n_gpu or 1))


def n_gpu_set() -> int:
    return int(_TP["P"])


def resolve(mode: Optional[str]) -> dict:
    """The mode's row: ``{mode, tier, levers, env, route, worker_base, variant, kernels, stage, attach, cli_route}`` (+ ``card_off`` = {lever:
    reason} on a card with CARD_DROPS for the mode, + ``off`` = {lever: reason} for the row's own ablation entries, + ``run_off`` =
    {lever: option} when this run takes levers off by name, set_run_drops, + ``tp_off`` = {lever: reason} at n_gpu > 1, set_n_gpu / TP_DROPS: every
    way the levers, their switch words and idle attachments leave the row, named); ``ValueError`` on an unknown mode."""
    m = (mode or DEFAULT_MODE).strip().lower()
    if m not in MODES:
        raise ValueError(f"unknown mode '{mode}': the modes are {'|'.join(MODE_NAMES)}")
    row = _tabled(m)
    for lever, why in list(_RUN_OFF.items()):
        take_off(row, lever, why, "run_off")
    conditional_riders(row, "run_off")                        # a run drop of the hoist (RUN_DROPS use_potentials) leaves the key gather no caller on a fused-atom row
    return row


def off_levers(row: dict) -> Dict[str, str]:
    """Every lever that left the row by name, in one map {lever: reason}: card drops, the row's ablation entries, the run's drops, the xP line's drops."""
    out: Dict[str, str] = {}
    for key in OFF_KEYS:
        out.update(row.get(key) or {})
    return out


def env_row(mode: str) -> Dict[str, str]:
    """The kits' activation row for `mode` (a copy)."""
    return dict(resolve(mode)["env"])


def levers(mode: str) -> List[str]:
    return list(resolve(mode)["levers"])


def worker_args(mode: str) -> Dict[str, str]:
    """The kit worker's arguments for `mode`: the probe arm's (WORKER_ARGS) with the row's own kernel state (`kernels`: the worker's
    switch, bz_worker_lev.py:31)."""
    return dict(WORKER_ARGS, kernels=resolve(mode)["kernels"])


# ---------------------------------------------------------------- the KERNELS census: what each route expects of upstream's accelerators ----------------------------------------------------------------
# The one statement of the expected word kind per accelerator and route (kernels.py is the reader and the guard; it never decides what a route
# expects). Upstream's two accelerators (kernels.ACCELERATORS: cuEquivariance triangle attention / triangle multiplication) follow the one flag
# `use_kernels`: ON on the stock CLI unless `--no_kernels` is among its arguments (main.py:1033-1037,1321; `--det 1` adds it),
# ON on every worker mode (the rows' "kernels": "on" -> the worker's `--kernels on`, bz_worker_lev.py:31,89,99). At n_gpu > 1 the
# row-sharded pair stack REPLACES the whole-tensor TriMul kernels by construction (rowpair.REPLACED_LEVERS "cuequivariance_trimul": the
# ring-streamed contraction serves every TriMul; the cuEquivariance triangle attention still serves the row blocks below the flash gate).
KERNELS_OFF_FLAG = "--no_kernels"                                     # upstream's own switch (main.py:1033-1037): the stock CLI's kernels-off word
KERNELS_TP_REPLACED = {"cueq_trimul": "replaced_by_rowpair"}          # rowpair.REPLACED_LEVERS["cuequivariance_trimul"] at n_gpu > 1 (tests/test_kernels.py locks the pair)


def kernels_expected(mode: str, n_gpu: int = 1, stock_args: Optional[List[str]] = None) -> Dict[str, str]:
    """{accelerator: 'engaged' | 'off-by-route:<reason>'} for a route: mode ``off`` reads the stock CLI's own arguments (``--no_kernels`` ->
    off-by-route:--no_kernels for both), a worker mode its row's kernel state (``worker_args(mode)["kernels"]``); ``n_gpu > 1`` names the
    accelerators the row-sharded pair stack replaces (KERNELS_TP_REPLACED)."""
    from .kernels import ACCELERATORS
    m = resolve(mode)["mode"]
    if m == "off":
        on = KERNELS_OFF_FLAG not in list(stock_args or [])
        reason = KERNELS_OFF_FLAG
    else:
        on = worker_args(m)["kernels"] == "on"
        reason = "worker--kernels=off"
    exp = {a: ("engaged" if on else f"off-by-route:{reason}") for a in ACCELERATORS}
    if on and int(n_gpu or 1) > 1:
        for a, why in KERNELS_TP_REPLACED.items():
            exp[a] = f"off-by-route:{why}"
    return exp


GUARD_ATTACHMENTS = ("templ",)   # attachments that apply no lever (the template guard): they never make a mode un-servable


def attachments(mode: str) -> List[str]:
    """Attach-time modules the worker process takes through boltz2_opt.worker_launch --attach (installed before the script's first import,
    applied right after the named trigger module executes): `templ` = boltz2_opt.templates.apply() after boltz.model.models.boltz2 (the
    template guard: every worker mode); `trimul` = boltz2_opt.trimul.apply() after boltz.model.layers.triangular_mult; `transition` =
    boltz2_opt.transition.apply() after boltz.model.layers.transition; `pairblock` = boltz2_opt.pairblock.apply() after
    boltz.model.layers.triangular_attention.attention (`triattn_exact` = boltz2_opt.triattn_exact.apply() after the same module, next);
    `pairfuse` = boltz2_opt.pairfuse after boltz.model.modules.trunkv2; `sampler` /
    `ditexact` = boltz2_opt.sampler / boltz2_opt.ditexact after boltz.model.modules.diffusionv2; `exactln` / `waste` / `msa` / `msa2` / `graph` /
    `atom` = their boltz2_opt adapters after boltz.model.models.boltz2; `xl` =
    boltz2_opt.big.apply() after boltz.model.models.boltz2 (the worker's own model import, bz_worker_lev.py:59)."""
    return list(resolve(mode).get("attach") or [])


def lever_attachments(mode: str) -> List[str]:
    """The attachments that install levers (attachments minus GUARD_ATTACHMENTS)."""
    return [a for a in attachments(mode) if a not in GUARD_ATTACHMENTS]


def routed_kernels(mode: str) -> List[str]:
    """The core-served kernels of the mode's worker (ROUTED_KERNELS; [] for the stock route and the modes that import none)."""
    return list(ROUTED_KERNELS.get(resolve(mode)["mode"], []))


def kernel_exports(name: str) -> Dict[str, str]:
    """The export parameters this tree passes for a routed kernel (KERNEL_EXPORTS; opt/-relative paths, resolved by stack / worker_launch)."""
    return dict(KERNEL_EXPORTS.get(name, {}))


def stage_files(mode: str) -> List[str]:
    """Carried kit files the worker process of `mode` needs in its working directory, plus the mode's worker base (opt/-relative)."""
    r = resolve(mode)
    if r["route"] != PINNED_ROUTE:
        return []
    return list(r["stage"]) + [r["worker_base"]]


def jit_cache_key() -> str:
    """torch<version sans local tag>-cu<CUDA version sans dot>-sm<compute capability digits> for the running box (e.g. torch2.12.0-cu130-sm90);
    'unknown' when torch or a CUDA device is absent. The Triton kernels (flash triangle attention, the fused cells) compile per box: the configs key the JIT cache by it."""
    try:
        import torch  # deliberate: only the config's cache-key derivation imports torch
        v = torch.__version__.split("+")[0]
        cu = (torch.version.cuda or "").replace(".", "")
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
        if not cu or cap is None:
            return "unknown"
        return f"torch{v}-cu{cu}-sm{cap[0]}{cap[1]}"
    except Exception:
        return "unknown"



# ---------------------------------------------------------------- the compile word ----------------------------------------------------------------
# Where stock compiles, modes inherit it untouched; a kit-added torch.compile lever ships off unless its gain is measured AND (its shapes are
# bucketed by stock with a warm Inductor/AOT cache shipped for every bucket, OR it holds under dynamic shapes). This kit: stock `boltz predict`
# builds Boltz2 with upstream's compile_* constructor flags at their False defaults and passes no compile option, and no row of this table adds
# a compile lever — nothing in any mode compiles. The sanctioned opt-out `--no-compile` is an alias of MODEL_OPT_LEVERS_OFF=compile (the
# uniform ablation word list the shared core's kernel packages read; this table's own by-name ablation is MODES[m]["off"]):
# the word joins this process's environment, so the worker, the stock subprocess and the core inherit one list. The ACTIVE / DRY-RUN lines print
# `compile=off:none_in_kit` (no lever to switch) or `compile=off:user` (the word given) — never `on` from this tree.
LEVERS_OFF_ENV = "MODEL_OPT_LEVERS_OFF"
COMPILE_WORD = "compile"
COMPILE_LEVERS: Tuple[str, ...] = ()                          # registry levers that would engage torch.compile: none in this kit (a future one lists itself here AND answers the word)


def levers_off_words(env: Optional[dict] = None) -> Tuple[str, ...]:
    """The words of MODEL_OPT_LEVERS_OFF (comma / whitespace separated), in order, duplicates dropped."""
    raw = (os.environ if env is None else env).get(LEVERS_OFF_ENV, "") or ""
    out: List[str] = []
    for w in raw.replace(",", " ").split():
        if w not in out:
            out.append(w)
    return tuple(out)


def add_levers_off_word(word: str) -> str:
    """Join `word` to MODEL_OPT_LEVERS_OFF in this process's environment (idempotent); returns the resulting value. `--no-compile` calls it with `compile`."""
    words = levers_off_words()
    if word not in words:
        words = words + (word,)
    os.environ[LEVERS_OFF_ENV] = ",".join(words)
    return os.environ[LEVERS_OFF_ENV]


def compile_word(env: Optional[dict] = None) -> str:
    """The `compile=` token of the ACTIVE / DRY-RUN lines: `off:user` when MODEL_OPT_LEVERS_OFF carries `compile` (`--no-compile`), else
    `off:none_in_kit` — this kit adds no compile lever and stock compiles nothing, so there is no `on` and nothing steps aside."""
    if COMPILE_WORD in levers_off_words(env):
        return "off:user"
    return "on" if COMPILE_LEVERS else "off:none_in_kit"

def env_mode() -> Optional[str]:
    """The mode named by BOLTZ2_OPT in this process's environment (None when unset/empty)."""
    v = os.environ.get("BOLTZ2_OPT", "").strip().lower()
    return v or None

