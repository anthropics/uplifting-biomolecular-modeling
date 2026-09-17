"""Modes, and the kit arm each one resolves to.

The kit's switch is the arm string `levers_ptx1.apply()` takes: `<trimul>[+lever...]` with trimul in stock|fast|exact and the lever words of
LEVERS below (`kit.lever_grammar()` reads both lists out of the kit file). The modes below are the only arms reachable through `pred`.

Package modes (`KIT_MODES`, the one place that says what a mode means — each entry carries its arm string; `MODES` / `DEFAULT_MODE` are the
one list and the one default every command reads):
  exact   the exact-class triangle multiplication and the exact-class levers of LEVERS (the fused triangle-attention block around the stock
          cuEquivariance core `gblock` + its provider word `triexact`, the pair transition `xtr`, the sampler CUDA graphs `sg`, the DiT pair-bias hoist `hoist`,
          `keep_pool`, `summary_hostidx`, `dit_attn_exact`, the template levers' exact words, ...). Tier 1: equal to stock bit for bit under
          the deterministic recipe (det.py) on the pinned stack; a lever without a build for the running card steps aside by name.
  fast    the fast-class triangle multiplication (Triton, bf16 MMA / fp32 accumulate) and the tolerance-class levers of LEVERS (the flash
          triangle-attention block `gflash` + `tricuda`, the fused LN+SwiGLU transitions `ttr`, the DiT / atom attention and fused sampler
          stacks, the fused MSA kernels, ...) plus the same `sg`, `hoist`, `keep_pool`, `summary_hostidx`. Tier 2: same error class as
          stock, not bit-equal. The package default (`DEFAULT_MODE`). exact / fast keep the sampler graphs + hoist up to the graph's token
          cap (graph_cap); use big for larger inputs.
  big   the memory mode: the `fast` arm without the base levers that hold memory through the item (big.BASE_LEVERS_OFF: the sampler
          CUDA graphs `sg` + `sampler_prep`, the DiT pair-bias hoist, the `keep_pool` allocator policy, the fused DiT stack `dit_fused` +
          `dit_lowp`), dropped at every size, plus the memory levers of big.py, sized once per process on the run's largest input item
          (big.compose) for its size gates — the upstream's torch TriangleMultiplication engaged above big.TRIMUL_GATE_TOKENS. Tier 2
          (fast-class numerics, within the band). `--n_gpu P` is its resource axis (ngpu.py). `resolve("big")` returns the arm so sized.
  off     stock: the upstream `protenix pred` in a clean subprocess (stock_pred.py); nothing from the kit on the path. A process of
          your own with PROTENIX_V1_OPT unset is stock too (the .pth installs no finder then).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from . import kit as K

MODES: Tuple[str, ...] = ("exact", "fast", "big", "off")
DETERMINISM: Tuple[Tuple[str, str], ...] = (          # one row per mode: what the mode's outputs are against stock's under the recipe (--det 1)
    ("exact", "bitwise"),                                #   tier 1: every output file equal to stock's under the deterministic recipe (the core's exactness label)
    ("fast", "band"),                                    #   tier 2: within stock's own seed-to-seed spread, not bit-equal
    ("big", "band"),                                   #   tier 2: the fast base + the memory levers (big.py) — fast-class numerics, within the band; the neutral levers are bit-equal by construction
    ("off", "stock"),                                    #   the stock path itself (not self-reproducing at --det 0; bit-reproducible under --det 1)
)
DEFAULT_MODE = "fast"                                   # the package default wherever a fast mode ships (the amortized warm steady-state is the timing that counts); exact / off by name
STOCK_ARM = "stock"


@dataclass(frozen=True)
class KitMode:
    mode: str
    arm: str                  # the string `levers_ptx1.apply()` takes
    line: str                 # the kit README's name for the line
    tier: Optional[int]       # 1 = equal to stock bit for bit under the deterministic recipe, 2 = same error class, None = stock
    promise: str
    memory_preset: Optional[Dict[str, object]] = None   # MEMORY_PRESET for the memory mode: the upstream's own inference settings, applied to the runner's configs
    triattn_floor: Optional[int] = None                 # a flash triangle-attention token floor (TRIATTN_FLOOR: None — the kit has none; kept for a mode that would carry one)


# The flash triangle-attention lever has NO kit size floor: `gflash` serves every item upstream's own statement hands the kernel whole
# (> 16 tokens — its small-input torch route, levers_ptx1.TRIATTN_GATE_TOKENS, an accounted `gate-off` state when every item is at or
# below it — and not row-chunked, <= levers_ptx1.TRIATTN_CEILING_TOKENS), and the shared core's triangle-attention provider names the
# attention core per (card, dtype, head_dim, heads, row length) through its tier words (`tricuda`; `triexact` on gblock). TRIATTN_FLOOR stays None: nothing is
# exported as PTX_TRIATTN_MIN_TOKENS and the report carries no FLOOR line.
TRIATTN_FLOOR: Optional[int] = None

# The memory mode (`big`): ONE composition — the `fast` base plus the memory levers of big.py (the upstream's own chunked pair path
# `chunk_pair` = MEMORY_PRESET below, the dead bond_mask dropped, the lean relative-position encoding, z_init parked on the host across the
# recycle seams, the diffusion conditioning and the confidence head in row blocks, the two release levers, the expandable-segments allocator
# (no CUDA-graph pool remains under the mode), and — above big.TRIMUL_GATE_TOKENS only — the upstream's torch TriangleMultiplication), with
# the base levers that HOLD memory through the item off at every size (big.BASE_LEVERS_OFF; the arm is KIT_MODES["big"].arm less those).
# The composition is sized once per process on the run's largest input item (big.compose: the command line's --input, or
# PROTENIX_V1_BIG_SIZE_N_TOKEN); `resolve("big")` returns the arm so sized. Guarantee: folds bigger on the same card, fast-class numerics
# (tier 2: inside the band, not bit-equal — the levers themselves are bit-equal or band by construction, big.TABLE), slower than fast where
# fast fits. MEMORY_PRESET is chunk_pair's setting: the smallest chunk of the upstream's own ladder (configs_base.py infer_setting: chunk_size 256
# with chunk_size_thresholds {1024: none, 1536: 512, 2048: 256, 2560: 128}) at every size with the thresholds off, set in place on
# `InferenceRunner.configs` after the stock runner is built (big.apply; runner/inference.py update_inference_configs mutates the same object per
# item, the values survive). The MEMORY line records the values applied and replaced; the BIG line and the activation report's big record the
# composition, the gates and the census.
MEMORY_PRESET: Dict[str, object] = {"infer_setting.chunk_size": 128, "infer_setting.dynamic_chunk_size": False}

KIT_MODES: Dict[str, KitMode] = {
    "exact": KitMode("exact", "exact+gblock+triexact+xtr+sg+hoist+keep_pool+summary_hostidx+dit_attn_exact+lazy_init+template_dedupe+sampler_prep+atom_attn_exact+tmpl_triatt+tmpl_xtr+tmpl_trimul_exact", "T1+hoist", 1, "equal to stock bit for bit under the deterministic recipe on the pinned stack"),
    "fast": KitMode("fast", "fast+gflash+tricuda+ttr+sg+hoist+keep_pool+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+sampler_prep+pfattn+opm_fused+pwa_fused+cond_dedupe+dit_fused+dit_lowp+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused", "T2+hoist", 2, "same error class as stock (bf16 MMA / fp32 accumulate kernels); not bit-equal", triattn_floor=TRIATTN_FLOOR),
    "big": KitMode("big", "fast+gflash+tricuda+ttr+sg+hoist+keep_pool+summary_hostidx+ditattn+ditattnfp16+atomattn+lazy_init+template_dedupe+tmpl_triatt+sampler_prep+pfattn+opm_fused+pwa_fused+cond_dedupe+dit_fused+dit_lowp+atom_fused+tmpl_trimul+tmpl_xtr+tmpl_pairfused", "big", 2, "folds bigger on the same card: the fast base without its sampler graphs, DiT hoist and kept allocator pool (off at every size) plus the memory levers (big.py); fast-class numerics (within the band), slower than fast where fast fits", MEMORY_PRESET, TRIATTN_FLOOR),
    "off": KitMode("off", STOCK_ARM, "stock", None, "the upstream package as pinned, nothing from the kit on the path"),
}

# The kit's levers (the names `levers_ptx1.apply()` takes), each with its numerics class and the kit file that implements it.
LEVERS: Dict[str, dict] = {
    "exact": {"class": "exact", "what": "the exact-class triangle multiplication (c_z = c_hidden = 128) served through the shared core's TriMul provider by the `exact` TIER word per (cc, precision, N-bucket, direction) key (kit 0.2.38: opt_core.kernels.trimul select(word='exact') names the row, triangle_multiplication(selection=...) serves it): `native_exact` where the pinned stack's table vouches it bit for bit against the stock kernel (bf16 trunk keys on H100 and A100 at opt_core >= 0.5.104), the cuEquivariance library row `cueq` at the fp32 / tf32 confidence-head keys; a key the provider refuses by name keeps upstream's statement (`stock:trimul:<refusal>`); no row word, no size floor, no kit-side kernel statement; LEVER `row=` / `served_by=` / `selections=` / `bind=tier:exact`", "file": "opt_core/kernels/trimul (the provider; rows native_exact / cueq)"},
    "fast": {"class": "tolerance", "what": "the fast-class triangle multiplication (c_z = c_hidden = 128) served through the shared core's TriMul provider by the `fast` TIER word per key (`big` under --mode big), kit 0.2.38: whatever row the pinned stack's table names (at opt_core >= 0.5.104: H100 `native` at the bf16 keys / `native:f32in` at the tf32 keys, A100 `v4`; `big` -> `v4` on H100, `native` on A100); a key the provider refuses by name keeps upstream's statement (`stock:trimul:<refusal>`); no row word, no size floor, no kit cell table; LEVER `row=` / `served_by=` / `selections=` / `bind=tier:<word>`", "file": "opt_core/kernels/trimul (the provider; rows native / native:f32in / v4)"},
    "gblock": {"class": "exact", "what": "the core's fused triangle-attention block around the STOCK cuEquivariance core: stock LayerNorm -> one-kernel q|k|v|g|bias prologue -> cuEquivariance attention -> one-kernel sigmoid-gate * o @ W_o epilogue (impl fpf)", "file": "opt_core/attn/pair_fused.py (impl fpf: opt_core/kernels/fpf_triatt_pro, fpf_triatt_epi; routed by name)"},
    "gflash": {"class": "tolerance", "what": "the core's fused triangle-attention block with the LayerNorm in the prologue kernel (impl lnl) around a flash attention core, at every row length upstream hands the kernel whole (> 16 tokens, not row-chunked: <= 2048) — no kit size floor; the attention core per key = `tricuda`'s provider tier word (LEVER `cores=`), the fused block's own default core (`core=default`) with that word off", "file": "opt_core/attn/pair_fused.py (impl lnl: opt_core/kernels/lnl_fused; core: opt_core.kernels.triattn by tier word under tricuda; routed by name)"},
    "tricuda": {"class": "tolerance", "what": "rides gflash: the block's attention core through the provider by its `fast` TIER WORD per key at call form mask_bias (the provider's `big` tier word under --mode big): whatever row its cells name on this card and stack — triattn_native / cuda_sm90a / k2b / flash … (bf16 tensor-core products, fp32 accumulation: one numerics class); a key the tier word refuses BY NAME (no cell for the class, no candidate admitted, no prebuilt for the stack) keeps the stock cuEquivariance statement inside the block, counted cueq:<word>", "file": "opt_core/kernels/triattn/ (tier word fast; selected by opt_core.kernels.triattn.select, routed by ptxfpf/levers_ptx1.py gflash_core)"},
    "triexact": {"class": "exact", "what": "rides gblock: the block's attention core through the provider by its `exact` TIER WORD per key at call form mask_bias: the exact-class row `triattn_exact` (a CUDA triangle attention whose output equals the library op's bit for bit) on a (card, torch, library) stack the core's cell table vouches for it, the library op itself (row `cueq`: the block's own stock cuEquivariance statement) BY NAME everywhere else — on the pinned library build no cell is vouched, so every key names `cueq` and the bytes are gblock's without the word; a key the tier word refuses by name keeps the stock statement inside the block, counted cueq:<word>", "file": "opt_core/kernels/triattn/ (tier word exact; selected by opt_core.kernels.triattn.select, routed by ptxfpf/levers_ptx1.py gblock_core)"},
    "xtr": {"class": "exact", "what": "the pair transitions (C=128, hidden 512) and the MSA transition (C=64, hidden 256) by the shared core's transition provider's `exact` TIER word, the stock LayerNorm output handed in: the provider's bitwise-vouched kernel row for this stack, size and row count serves (cc 9.0: the fused SwiGLU pair transition, the hidden never touches HBM); a cell whose exact tier is the stock statement (the MSA transition's), a stack or size without a vouch, or a row count outside the card's served rows (opt_core.attn.pair_fused exact_rows: cc 8.0) keeps the stock module BY NAME; single-representation transitions and the diffusion conditioning's c=128 x 256 transitions stay stock (by name)", "file": "opt_core/kernels/transition (select('exact') / transition(); routed by name)"},
    "ttr": {"class": "tolerance", "what": "the pair (C=128 x 512) and MSA (C=64 x 256) transitions by the shared core's transition provider's `fast` TIER word (`big` under --mode big): the cell table's measured winner per card, shape and size (cc 9.0: the one-kernel v2 row; cc 8.0: its measured v2 / v1 cells); a refusal by name keeps the stock module; single-representation transitions, the template pairformer's c=64 x 128 (tmpl_xtr's) and the diffusion conditioning's c=128 x 256 transitions stay stock (by name)", "file": "opt_core/kernels/transition (select('fast'|'big') / transition(); routed by name)"},
    "sg": {"class": "exact", "what": "CUDA-graph capture/replay of the diffusion denoiser step (GraphedDenoiseLoop)", "file": "lib/kit112_src/infopt_graphs/"},
    "sampler_prep": {"class": "exact", "what": "host path of the graphed diffusion sampler (rides `sg`; exact + fast; big composes no sampler graph): pinned non-blocking upload of the per-step rotation, a shape-keyed graph cache with on-device comparison of the integer / mask feature values, one warm-up pass, the poison self-test once per process, the previous entry's static buffers released before a new capture and its pool re-used when the new signature is no larger (else returned to the device first), vectorised per-step scalars — no kernel, RNG draw or graph changes (exact by construction; `--det 1` byte identity); all six parts or none: the LEVER line's parts= word must read rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec and poison= ok, else the lever is a fallback by name", "file": "lib/kit112_src/infopt_graphs/ (protenix/sampler_prep.py + graphed.py sites)"},
    "hoist": {"class": "exact", "what": "DiT pair-bias hoist: the pair-bias projection computed once per sample outside the 200-step loop (no uninstall: one hoist arm per process)", "file": "lib/kit112_src/dit_hoist.py"},
    "lazy_init": {"class": "exact", "what": "start-up lever: the model's random parameter initialisation (torch.nn.init's and protenix's truncated-normal / lecun / he / glorot helpers, about a minute of CPU per process for the 368 M parameters) skipped while the stock runner constructs Protenix(configs) — runner.inference.InferenceRunner.load_checkpoint() then load_state_dict(strict=True) overwrites every parameter and persistent buffer, so no loaded value differs (PTX_LAZY_INIT=recheck constructs both ways in one process and compares sha256 of every state tensor: N/N); nothing at predict time changes", "file": "lib/ptx1_lazy_init.py (installed by opt/protenix_v1_opt/stack.py before the runner is built)"},
    "keep_pool": {"class": "exact", "what": "caching-allocator policy: the stock torch.cuda.empty_cache() calls inside the model forward (protenix 1.1.0 confidence.py:204/:233/:347) are counted no-ops so the blocks the diffusion sampler freed serve the confidence head without a synchronous release and re-growth from the driver; the per-item runner release (runner/inference.py:497/:510) and every kit release pass through; no tensor value depends on it", "file": "lib/ptx1_keep_pool.py"},
    "template_dedupe": {"class": "exact", "what": "template-embedder distinct-template evaluation: the featuriser always assembles 4 template slots (use_template off: slot 0 = restype gap, slots 1-3 zero), the 2-block c=64 Pairformer runs once per DISTINCT slot per recycle and the stock accumulation adds the representative's tensor for the duplicates (u = (((0+v0)+v1)+v1)+v1): bitwise to the stock loop under the deterministic recipe", "file": "ptxfpf/ptx1_templ.py"},
    "tmpl_triatt": {"class": "exact", "what": "the template Pairformer's triangle attention (c_in=64, 4 heads x 32) through the trunk's fused triangle-attention block, in the construction of the trunk lever it rides: with gblock (exact) impl fpf on the stock LayerNorm's output and the exact core — bit-identical to stock (det-1 bytes equal on cc 9.0 and 8.0), the class named here; with gflash (fast, big) the lnl LN+bias and gate cells at (64,)/(128,), one q|k|v|g GEMM and gflash's attention core (the provider's tier word per key; tolerance class, gflash's); routed once per shape from the core's cell table, the stock statement by name where no cell serves", "file": "ptxfpf/ptx1_templ.py (route) + ptxfpf/levers_ptx1.py (_tri_forward)"},
    "tmpl_trimul": {"class": "tolerance", "what": "the template Pairformer's triangle multiplication (outgoing + incoming; c_z 64, c_hidden 128, where the stock takes its torch path) through the core's triangle-multiplication provider by the MODE's tier word (opt_core.trimul.by_word over opt_core.kernels.trimul: `fast` under --mode fast, `big` under --mode big = the provider's measured winner of the call's cell on this card); a selection of the stock's own class idles by name (stock:tmpl_trimul:<row>:<why>), refusals by name (stock:tmpl_trimul:<row>:<kind>); no kit-side size floor or cell table", "file": "ptxfpf/ptx1_templ.py (route) + ptxfpf/levers_ptx1.py (_trimul_forward)"},
    "tmpl_trimul_exact": {"class": "exact", "what": "the same template triangle multiplication through the provider's `exact` tier word (--mode exact): an exact-class kernel row the provider vouches bitwise on this stack serves the call; where it names the stock op the lever idles by name (stock:tmpl_trimul:<row>:<why>) and the stock torch path answers — bytes ≡ --mode off either way", "file": "ptxfpf/ptx1_templ.py (route) + ptxfpf/levers_ptx1.py (_trimul_forward)"},
    "tmpl_pairfused": {"class": "tolerance", "what": "the template triangle-attention block's prologue / epilogue in impl fpf with the LayerNorm fused (pair_fused fpf prologue / epilogue cells (64, 4, 32), opt_core >= 0.5.28.0) for the calls tmpl_triatt routes under gflash, instead of impl lnl; kept on lnl by name where the fpf cells do not serve", "file": "ptxfpf/ptx1_templ.py (triatt_impl) + ptxfpf/levers_ptx1.py (_tri_forward)"},
    "tmpl_xtr": {"class": "exact", "what": "the template Pairformer's pair transition (c 64, n 2) through the core's transition provider by the mode's tier word first (opt_core.kernels.transition: exact | fast | big, family rows; a refusal by name hands the call on), else through the core's fused transition in the exact construction of `xtr` (impl fpf, the stock LayerNorm's output handed in; pair_fused transition cell (64, 128), opt_core >= 0.5.28.0); a shape the table does not serve or a refused call keeps the stock statement by name", "file": "ptxfpf/ptx1_templ.py (route) + ptxfpf/levers_ptx1.py (_transition_forward)"},
    "summary_hostidx": {"class": "exact", "what": "summary-confidence bookkeeping on the host: the chain / chain-pair token and atom index lists from ONE device-to-host copy of (asym_id, has_frame) and (atom_to_token_idx, atom_is_polymer) per item, uploaded as one int64 blob; per sample the same value-producing GPU ops as stock (integer-index gathers of the identical elements in the identical order, the stock reductions on bitwise-equal inputs) — removes the per-chain .item() / boolean-mask / pageable-copy synchronisations of protenix 1.1.0 sample_confidence.compute_full_data_and_summary (thousands per item, growing with chains²); a signature drift refuses by name, a function already owned by the row-sharded line steps aside by name", "file": "lib/ptx1_summary_host.py"},
    "ditattn": {"class": "tolerance", "what": "the 24 DiffusionTransformer blocks' token attention with pair bias (16 heads x 48, the samples batched, the pair bias [1,16,N,N] shared by the samples — the hoist's cached bias included) on ONE fused Triton flash-attention launch per call (q/k/v/g strided views of the stock projections, fp32 online softmax / accumulation, the output gate fused in the epilogue) instead of scaled_dot_product_attention + transposes + the gate chain; 3xTF32 operands (fp32-faithful; tolerance only because the reduction order differs); cell per card apb_ptx1.CELLS['dit'][cc]['fp32']; a call or card outside the cells steps aside by name (stock:<word> / named)", "file": "ptxfpf/apb_ptx1.py, opt_core/kernels/apb/fpf_apb/ (row fpf_apb: apb_triton.py, fpf_apb 0.2.2; bound by row word since kit 0.2.21, before it vendored under lib/ byte-identical)"},
    "ditattnfp16": {"class": "tolerance", "what": "PRECISION lever on ditattn: the same kernel with fp16 tensor-core operands (stock's fp32 q, k, v pre-cast to fp16 by the cell's precast form; P in fp16), bias / logits / softmax statistics / accumulation fp32; rides ditattn (no install of its own: ditattnfp16 without ditattn is named and installs nothing)", "file": "ptxfpf/apb_ptx1.py, opt_core/kernels/apb/fpf_apb/ (row fpf_apb: apb_triton.py, variant fp16)"},
    "atomattn": {"class": "tolerance", "what": "the DiffusionModule's atom encoder (3) + decoder (3) blocks' 32-query x 128-key local-window attention with pair bias on one Triton launch per call (windows and the key-range rule in-kernel, TF32-rounded operands, fp32 softmax / accumulation, the gate fused) instead of pad + unfold + the -1e10 mask + the chunked fp32 attention; the input embedder's atom encoder keeps stock", "file": "ptxfpf/apb_ptx1.py, opt_core/kernels/apb/fpf_apb/ (row fpf_atom: atom_triton.py)"},
    "dit_attn_exact": {"class": "exact", "what": "the DiffusionTransformer's pair-bias attention (protenix.model.modules.primitives._attention -> fp32 torch SDPA, 16 heads × 48, bias broadcast over the 5 samples; the 24 DiT blocks × 200 steps) on the shared core's prebuilt sm_90 CUDA kernel, bit-identical to the memory-efficient SDPA kernel it replaces (manifest + sha256 + a 3-case torch.equal load-time check in every process); the sampler graph's warm-up / capture calls are 4-D and served (so the captured denoiser step replays the kernel), protenix 1.1.0's eager 5-D calls and other head widths take the stock statement, counted by reason; a card or stack without a prebuilt (cc 8.0 today) steps aside by name — exact only: the fast modes' DiT attention slot is ditattn's", "file": "opt_core/kernels/apb/dit_exact (row dit_exact, dit_exact 1.0.0 + its sm_90 prebuilt; bound by row word since kit 0.2.21, before it vendored under opt_core/kernels/apb/dit_exact byte-identical)"},
    "pfattn": {"class": "tolerance", "what": "the trunk PairformerStack's (48 blocks x N_cycle) and the confidence pairformer's (4 blocks) single attention with pair bias through the shared core's attention-with-pair-bias provider (opt_core.kernels.apb) by the mode's TIER word: the provider's bias-producer row over z (LayerNorm + the 16-head projection, head-major planes) and its attention-core row (shared bias, sigmoid gate) on stock's bf16 q/k/v/g projections, then stock's linear_o; rows and cells are the provider's per card, a row's refusal or the boundary's library row is the module's own statement for that call, by name (ptxfpf/trunk2_ptx1.py)", "file": "ptxfpf/trunk2_ptx1.py, lib/protenix_fpf_apb/ (pf_triton.py, apb_triton.py)"},
    "opm_fused": {"class": "tolerance", "what": "the MSA module's four OuterProductMean forwards: fused LayerNorm(m) + both hidden projections in one pass over m, stock's own einsum GEMM per row chunk, then linear_out on the einsum's native layout with the bias and the 1/norm in the kernel's epilogue (no flatten / permute copies); mask / fp16 / fp32 (no-autocast) calls keep the stock statement by name", "file": "ptxfpf/trunk2_ptx1.py, lib/protenix_fpf_msa/ (msa_triton.py)"},
    "pwa_fused": {"class": "tolerance", "what": "the MSA module's three MSAPairWeightedAveraging forwards (per 2048-row MSA chunk): fused LayerNorm(m) + the v|g projections writing v head-major, the pair weights from the one-pass LayerNorm(z)+linear_z producer + softmax, the weighted average as one cuBLAS bmm, and sigmoid(g) * o -> linear_out in one kernel, instead of LayerNorm + three Linears + two permuted einsum copies + the gate chain; fp32 (no-autocast) calls keep the stock statement by name", "file": "ptxfpf/trunk2_ptx1.py, lib/protenix_fpf_msa/ (msa_triton.py), lib/protenix_fpf_apb/ (pf_triton.py)"},
    "cond_dedupe": {"class": "tolerance", "what": "the diffusion conditioning evaluated ONCE per denoiser step: protenix 1.1.0's sampler hands the denoiser the step's noise level as one scalar expanded over the 5 samples (stride 0), so stock computes the identical [N, c_s] DiffusionConditioning output (noise embedding, LayerNorm + Linear, two transitions) five times and every consumer downstream five times too; the lever runs DiffusionConditioning.forward on t_hat[..., :1] when the sample dimension is an expanded scalar (decided from strides, no device read; capture-safe inside `sg`) and returns s with a sample dimension of 1 that every stock consumer broadcasts — a t_hat that is not an expanded scalar takes the stock rows, counted (never refused); tolerance class by rule (the same values through GEMMs of another row count); a card without a cell (cc 8.0 today) steps aside by name", "file": "ptxfpf/ditfast_ptx1.py, lib/protenix_fpf_ditfast/ (cond_dedupe.py; the carried protenix_fpf_ditfast package)"},
    "dit_fused": {"class": "tolerance", "what": "the sampler's 24-block token DiffusionTransformer as ONE fused forward: per block an AdaLN row kernel, one q|k|v|g GEMM, the fused pair-bias attention protenix_fpf_apb.dit_apb (ditattn's kernel and measured cell — REQUIRED: without ditattn engaged the lever installs nothing and is named), o GEMM, [gate*x + fp32 residual + next AdaLN] row kernel, one a1|a2 GEMM, SwiGLU row kernel, b GEMM, [gate + residual + the next block's AdaLN] kernel; the conditioning side of all blocks as two GEMMs per call; the 24 pair biases produced once per item into the DiT hoist's slots (per call without `hoist`); Triton row kernels + cuBLAS, capture-safe inside `sg`; a card without a cell (cc 8.0 today) steps aside by name", "file": "ptxfpf/ditfast_ptx1.py, lib/protenix_fpf_ditfast/ (dit_fast.py, kernels.py), lib/kit112_src/dit_hoist.py (the bias slots)"},
    "dit_lowp": {"class": "tolerance", "what": "PRECISION lever on dit_fused (word fp16): the fused token stack's a-path activations / GEMM operands and attention operands in fp16 with fp32 accumulation; LayerNorm and softmax statistics, the residual stream and the conditioning fp32 (ditattnfp16's error class extended to the block's GEMMs); rides dit_fused (no install of its own: dit_lowp without dit_fused is named and installs nothing)", "file": "ptxfpf/ditfast_ptx1.py, lib/protenix_fpf_ditfast/ (dit_fast.py: PTX_DIT_LOWP)"},
    "atom_fused": {"class": "tolerance", "what": "both 3-block atom transformers of the DiffusionModule (atom_attention_encoder / decoder .atom_transformer: c_atom 128, 4 heads x 32, 32x128 local windows) as fused stacks: entry AdaLN kernel, one q|g and one k|v GEMM per block, the fused local attention protenix_fpf_apb.atom_apb (atomattn's kernel — REQUIRED, as dit_fused requires ditattn), o GEMM, row kernels, one a1|a2 GEMM, SwiGLU, b GEMM; every step-invariant operand (the per-block conditioning with sigmoid applied, the local pair bias) produced once per item through the hoist's slots; TF32-order differences only (tolerance); capture-safe inside `sg`; a card without a cell (cc 8.0 today) steps aside by name", "file": "ptxfpf/ditfast_ptx1.py, lib/protenix_fpf_ditfast/ (atom_fast.py, atom_kernels.py)"},
    "atom_attn_exact": {"class": "exact", "what": "the atom transformer's 32-query x 128-key local-window attention (AttentionPairBias.attention on the 3 encoder + 3 decoder atom blocks: 4 heads x 32, fp32; x 200 steps) through the shared core's attention-with-pair-bias face by the `exact` TIER WORD: opt_core.kernels.apb.select(cc, fp32, atom_h4d32w32x128, N, word=exact) names the row — a kernel row the core vouches byte-exact on this stack is served through apb.atom_attention at those modules (re-selected per call at the call's size); when the word names the stock statement (row sdpa_gather, class stock: every card of opt_core 0.5.117) nothing is installed and the upstream op serves, BY NAME (`exact_word:<row>` — bytes are --mode off's by construction); no kit cell table, no card list — exact only: the fast modes' slot is atomattn's / atom_fused's (asked beside them it steps aside by name)", "file": "ptxfpf/ditfast_ptx1.py (install_atom_exact_by_word over opt_core.kernels.apb)"},
}


def check_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    m = str(mode).strip().lower()
    if m not in MODES:
        raise ValueError(f"unknown mode {mode!r} (expected one of {'|'.join(MODES)})")
    return m


@dataclass(frozen=True)
class Resolution:
    mode: str
    arm: str
    trimul: str
    levers: Tuple[str, ...]
    tier: Optional[int]
    line: str
    memory_preset: Optional[Dict[str, object]] = None   # KitMode.memory_preset (the memory mode), else None
    triattn_floor: Optional[int] = None                 # KitMode.triattn_floor where the arm carries `gflash`, else None
    ablated: Tuple[str, ...] = ()                       # MODEL_OPT_LEVERS_OFF (ablation.py): the mode's levers withheld from this run, request order; () = the mode as shipped
    mode_arm: Optional[str] = None                      # the mode's own arm before the ablation (== arm when nothing is ablated)
    capped: Tuple[str, ...] = ()                        # the sampler-graph levers this run's largest input put above the graph's token cap (graph_cap): they step aside by name
    graph_cap: Optional[Dict[str, object]] = None       # {cap, source, memory_mib, n_token, n_token_source} when the mode carries a graph lever and the run was sized, else None


# The graphed diffusion sampler's token cap. `sg` (the denoiser-step CUDA graph: its private allocator pool stays live through the confidence
# head) and `hoist` (the 24 DiT blocks' pair biases cached for every step: 24 x 16 x N^2 fp32 values) are exact/fast's speed levers whose
# memory grows with N^2 on top of the trunk's, so a run whose largest input (big.size_of_run: PROTENIX_V1_BIG_SIZE_N_TOKEN, else the
# --input polymer estimate) exceeds the mode's cap resolves the arm without them, BY NAME: `capped=<levers>@<cap>` on the ARMED / ACTIVE
# lines and `LEVER … state=skipped reason=above_cap gated_by=graph_cap:<cap> n=<tokens>` per lever, exit 0 — instead of an out-of-memory
# exit. (`dit_attn_exact` is not a graph lever: it serves the eager sampler's calls too.) The cap is keyed on the mode and on the device's
# total memory (nvidia-smi): SAMPLER_GRAPH_MAXTOK at and above SAMPLER_GRAPH_MEM_MIB, SAMPLER_GRAPH_MAXTOK_SMALL below it (half the tokens =
# a quarter of the N^2 terms). A caller's PTX_SAMPLER_GRAPH_MAXTOK wins for every mode (tokens; 0 = no cap). An unsized run (no --input to
# estimate, e.g. `check`) applies no cap. big composes neither lever: nothing to cap.
GRAPH_CAP_LEVERS = ("sg", "hoist", "sampler_prep")
SAMPLER_GRAPH_MAXTOK_ENV = "PTX_SAMPLER_GRAPH_MAXTOK"
SAMPLER_GRAPH_MAXTOK = {"exact": 1536, "fast": 1999}          # devices at or above SAMPLER_GRAPH_MEM_MIB: the largest input each mode engages its graph levers for (they also carry sampler_prep)
SAMPLER_GRAPH_MAXTOK_SMALL = {"exact": 768, "fast": 999}      # devices below SAMPLER_GRAPH_MEM_MIB: half the tokens
SAMPLER_GRAPH_MEM_MIB = 64 * 1024
CAP_REASON = "above_cap"


def sampler_graph_maxtok(mode: str, memory_mib: Optional[int]) -> int:
    """The sampler-graph token cap of `mode` (exact | fast; another mode: 0 = none) on a device with `memory_mib` MiB of memory (None: no device
    probed -> the large-device cap)."""
    table = SAMPLER_GRAPH_MAXTOK if memory_mib is None or int(memory_mib) >= SAMPLER_GRAPH_MEM_MIB else SAMPLER_GRAPH_MAXTOK_SMALL
    return int(table.get(mode, 0))


def device_memory_mib() -> Optional[int]:
    """Total memory of GPU 0 in MiB from nvidia-smi (opt_core.gates.nvidia_smi_probe; no torch import), None when no card answers
    (the probe itself never raises: it reports `probe=<why>` with memory_mib None -> the large-device cap; the run names a missing GPU)."""
    from opt_core import gates as G
    mem = G.nvidia_smi_probe(keys=("memory_mib",)).get("memory_mib")
    return int(mem) if mem is not None else None


def graph_cap(mode: str, environ=None) -> Dict[str, object]:
    """{cap: int (0 = none), source: environment|device, memory_mib}: the caller's PTX_SAMPLER_GRAPH_MAXTOK, else `mode`'s device-keyed cap."""
    environ = os.environ if environ is None else environ
    raw = str(environ.get(SAMPLER_GRAPH_MAXTOK_ENV, "") or "").strip()
    if raw:
        try:
            cap = int(raw)
        except ValueError:
            raise RuntimeError(f"{SAMPLER_GRAPH_MAXTOK_ENV}={raw!r}: a token count (integer >= 0; 0 = no cap)") from None
        if cap < 0:
            raise RuntimeError(f"{SAMPLER_GRAPH_MAXTOK_ENV}={raw!r}: a token count (integer >= 0; 0 = no cap)")
        return {"cap": cap, "source": "environment", "memory_mib": None}
    mem = device_memory_mib()
    return {"cap": sampler_graph_maxtok(mode, mem), "source": "device", "memory_mib": mem}


def _apply_graph_cap(res: "Resolution", kit_dir: Optional[str], environ) -> "Resolution":
    """The resolution with the graph levers withheld by name when the run's largest input is above the mode's cap (see GRAPH_CAP_LEVERS)."""
    carried = tuple(lv for lv in GRAPH_CAP_LEVERS if lv in res.levers)
    if not carried:
        return res
    from . import big as B                             # the run's size, as the memory line sizes itself (one estimate for the kit)
    try:
        size = B.size_of_run(environ)
    except B.BigError as e:
        raise RuntimeError(f"{e}") from None
    cap = graph_cap(res.mode, environ)
    n = size.get("n_token")
    facts = dict(cap, n_token=n, n_token_source=size.get("source"))
    if n is None or not cap["cap"] or int(n) <= int(cap["cap"]):
        return replace(res, graph_cap=facts)
    capped = tuple(lv for lv in GRAPH_CAP_LEVERS if lv in res.levers)
    arm = "+".join(w for w in res.arm.split("+") if w not in capped)
    trimul, levers = K.parse_arm(arm, kit_dir)
    return replace(res, arm=arm, trimul=trimul, levers=levers, capped=capped, graph_cap=facts)


def resolve(mode: str, kit_dir: Optional[str] = None, environ=None) -> Resolution:
    """The kit arm of a package mode, validated against the kit's own grammar (`kit.parse_arm`). ``MODEL_OPT_LEVERS_OFF`` (ablation.py) is
    applied here, so every route reads one resolution: the named arm words leave the arm string, the named memory levers ride on `.ablated` for
    the big line's selection (big.arm); a name the mode does not compose is a RuntimeError worded by name (the activation's NOT ACTIVE reason)."""
    from . import ablation as A                          # the ablation switch: read once per resolution from the environment
    m = check_mode(mode)
    km = KIT_MODES[m]
    memory_levers: Tuple[str, ...] = ()
    if m == "big":                                     # the composition sized on this process's input (big.compose): the base arm minus the levers off by property
        from . import big as B                         # big imports modes: read at call time
        try:
            comp = B.compose() if environ is None else B.compose(environ)
        except B.BigError as e:
            raise RuntimeError(f"{e}") from None
        km = KitMode(km.mode, comp["arm"], km.line, km.tier, km.promise, km.memory_preset,
                     km.triattn_floor if "gflash" in comp["arm"].split("+") else None)
        memory_levers = tuple(B.LINE)
    trimul, levers = K.parse_arm(km.arm, kit_dir)
    for lv in levers:
        if lv not in LEVERS:
            raise RuntimeError(f"mode {m}: lever {lv!r} has no registry entry")
    if km.triattn_floor is not None and "gflash" not in levers:
        raise RuntimeError(f"mode {m}: the arm {km.arm!r} has no gflash but triattn_floor={km.triattn_floor!r}")
    names = A.requested(environ)
    if not names:
        return _apply_graph_cap(Resolution(m, km.arm, trimul, levers, km.tier, km.line, km.memory_preset, km.triattn_floor, (), km.arm), kit_dir, environ)
    trimuls, all_levers = K.lever_grammar(kit_dir)
    known = tuple(t for t in trimuls if t in A.TRIMUL_WORDS) + tuple(all_levers) + _memory_lever_names()
    try:
        A.validate(m, names, km.arm, memory_levers, known)
    except A.AblationError as e:
        raise RuntimeError(str(e)) from None
    arm = A.reduce_arm(km.arm, names)
    trimul, levers = K.parse_arm(arm, kit_dir)
    floor = km.triattn_floor if "gflash" in levers else None      # an ablated gflash takes its floor with it (nothing exports PTX_TRIATTN_MIN_TOKENS for a lever that is not applied)
    return _apply_graph_cap(Resolution(m, arm, trimul, levers, km.tier, km.line, km.memory_preset, floor, tuple(names), km.arm), kit_dir, environ)


def _memory_lever_names() -> Tuple[str, ...]:
    """The memory mode's lever names (big.LINE) for the ablation's `known` universe — a memory lever named under exact / fast is refused as
    `not a lever of mode <m>` rather than `unknown`."""
    from . import big as B
    return tuple(B.LINE)
