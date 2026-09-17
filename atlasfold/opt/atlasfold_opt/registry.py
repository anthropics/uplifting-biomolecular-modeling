"""LEVERS — what each lever id of modes.MODES is: its class (``cls``: exact | fast), the strategy id it reports under (a shared catalogue id
``F<k>.<name>`` where one exists, else LOCAL.atlasfold.<name>), the module that installs it and the provider it binds (where it binds one), the
stock attribute it patches (site = module:attr, file:lines of upstream v1.0.0 992067e), the fallback reasons its ledger EXPECTS (fallbacks for a
reason outside the list refuse the lever's gate at exit: exit 3) and ``what`` it does, in one paragraph."""
from typing import Dict

KIT = "atlasfold"
TU = "atlasfold.model.network.primitives.triangle_update"

LEVERS: Dict[str, dict] = {
    "alloc_expandable": dict(
        cls="exact", strategy="F7.expandable_segments", module="atlasfold_opt.hooks.alloc_expandable", provider="opt_core.mem.torch_alloc.write_conf (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: the live allocator via torch.cuda.memory._set_allocator_settings + os.environ for workers)",
        site="the kit process's CUDA caching allocator (a policy: no model statement is touched)",
        expected=("AFO_ALLOC_EXPANDABLE=0", "user_conf:", "graph_lever_in_row:", "write:"),
        what="torch's caching allocator with expandable segments (allocator setting: expandable_segments): reserved memory tracks allocated across the trunk / sampler / confidence phases instead of caching blocks the next phase cannot reuse — exact row peak RESERVED -2.6 / -7.2 / -6.5..-8.2 GiB at 512 / 896 / 1280 at fwd_s +-0.2 %, allocated -0.1 GiB, bytes unchanged; beside a CUDA-graph lever in the row (fast's denoiser_graph) it steps aside BY NAME (graph_lever_in_row:denoiser_graph — the core's default; AFO_ALLOC_EXPANDABLE=graphs composes them), so exact and big carry it; an operator's own PYTORCH_CUDA_ALLOC_CONF is never overwritten (present -> reported; another value -> state=skipped user_conf, installed and inert); AFO_ALLOC_EXPANDABLE=0 / MODEL_OPT_LEVERS_OFF=alloc_expandable keep the stock policy",
    ),
    "trimul_exact": dict(
        cls="exact", strategy="F2.trimul", provider="opt_core.trimul.by_word(<the tier word `exact`>) over opt_core.kernels.trimul: the provider's exact cell per (cc, precision, c_z, N bucket, direction) — an exact-class row only where the provider holds its bitwise record on this stack, else its stock cuEquivariance row `cueq` BY NAME (cc 9.0 / 8.0, c_z 128, cuequivariance 0.10: `cueq` today); AFO_TRIMUL_EXACT_WORD = a developer row word (exact-class / stock rows only)",
        site=f"{TU}:TriangleMultiplicationOutgoing.forward + TriangleMultiplicationIncoming.forward (triangle_update.py L141-201 / L228-290; the cuequiv branch)",
        expected=("mode_stock", "backend_torch", "device:", "rank:", "cueq:", "torch_math:", "tmk3_exact:", "native_exact:", "tx_sm90a_exact:", "ef2_cueq_tiles:"),   # + every other provider row's `<row>:` refusal (hooks/trimul.expected_words over ROW_NAMES)
        what="the stock cuequiv TriangleMultiplication served through the shared TriMul provider by the tier word `exact`: no kit floor, table or pin — small N, packed batches and cards without a cell are the provider's decisions, named on the LEVER line (`impl=trimul:<row>@exact/<cell>`); a call a row refuses by name runs the engine's own forward (the stock op), counted",
    ),
    "trimul_v4": dict(
        cls="fast", strategy="F2.trimul", provider="opt_core.trimul.by_word(<the tier word of the mode: `fast` | `big`>) over opt_core.kernels.trimul: the cell's fastest admitted row per (cc, precision, c_z, N bucket, direction) on this stack (cc 9.0 / c_z 128 / bf16: `native` under fast, `v4` under big; cc 8.0: `v4` under fast, `native` under big — the provider's table, not the kit's); AFO_TRIMUL_WORD = a developer row word",
        site=f"{TU}:TriangleMultiplicationOutgoing.forward + TriangleMultiplicationIncoming.forward",
        expected=("mode_stock", "backend_torch", "device:", "rank:", "v4:", "native:", "tmk3_fast:", "tx_sm90a:", "esm_v5_fwd:", "esm_v61:", "esm_shapes:", "cueq:", "torch_math:"),   # + every other provider row's `<row>:` refusal (hooks/trimul.expected_words over ROW_NAMES)
        what="the fast-class TriangleMultiplication served through the shared TriMul provider by the tier word of the mode (`fast` in fast, `big` in big): LEVER line `impl=trimul:<row>@<word>/<cell>`; a row that cannot serve a call names why (`<row>:<kind>`, e.g. `v4:n<101`, `tx_sm90a:no_prebuilt:<abi>`) and the engine's own forward (the stock cuEquivariance TriMul) serves that call, counted",
    ),
    "flash_triattn": dict(
        cls="fast", strategy="F1.flash_triattn", module="atlasfold_opt.hooks.triatt", provider="opt_core.kernels.triattn (the MODE's tier word: fast | big; the measured cell's row per cc, dtype, D, H, N, call form)",
        site=f"{TU}:cueq_tri_attn (L94-116; called by TriangleAttentionStartingNode/EndingNode.forward L483-491)",
        expected=("stock_row:", "rank", "no_calls", "no_cell:", "no_row:", "unknown_word:", "dtype_", "arch:", "head_dim_", "single_head", "no_candidate_admitted"),
        what="the stock triangle-attention call through opt_core.kernels.triattn by the mode's tier word (q/k/v [B,L,4,L,32], bias [B,1,4,L,L] fp32, mask [B,L,1,1,L]; the provider serves triattn_native / cuda_sm90a / k2b / flash / ... per measured cell and card and names its step-asides); serves the pair stacks below triatt_block's floor (the fused block takes N>=512)",
    ),
    "triatt_block": dict(
        cls="fast", strategy="LOCAL.atlasfold.triatt_block", module="atlasfold_opt.hooks.triatt_block", provider="opt_core.attn.pair_fused impl=fpf core=<triattn_core pick | default> ln=fused",
        site=f"{TU}:TriangleAttentionStartingNode.forward + TriangleAttentionEndingNode.forward (triangle_update.py L314-399 / L425-515; the whole statement LN -> qkv|g|bias -> attention -> gate -> out)",
        expected=("backend_torch", "training", "dtype:", "rank", "c:", "below_min_tokens", "cpu", "no-cell:"),
        what="opt_core.attn.pair_fused.tri_attn_block(residual=False): ONE prologue kernel (LN -> q|k|v|g|pair bias, bf16 weights packed once per module; the ending node reads z^T by address math and gets the transposed pair mask as a contiguous [B,L,1,1,L] key mask, counted mask_copies=) -> the core's flash triangle attention -> ONE epilogue kernel (sigmoid(g)*o @ W_o); bf16 z at c_z=128, N>=512 (min_tokens=512); cells = the core table's fpf prologue/epilogue rows at (128,4,32) resolved once at install (LEVER cells=/cell_rows=)",
    ),
    "triattn_core": dict(
        cls="fast", strategy="LOCAL.atlasfold.triattn_core", module="atlasfold_opt.hooks.triattn_core", provider="opt_core.kernels.triattn through opt_core.attn.pair_fused core='tier:<mode word>' (the measured cell per cc, dtype, D, H, keys, mask form)",
        site=f"{TU}:TriangleAttentionStartingNode.forward + TriangleAttentionEndingNode.forward THROUGH triatt_block (hooks/pair_cells.core_pick: the attention core of the block's served calls; no patch site of its own)",
        requires=("triatt_block",),
        expected=("index_2p31", "no_calls"),
        what="the fused block's attention core bound to opt_core.kernels.triattn by the MODE's tier word (attn.pair_fused core='tier:fast' under fast, 'tier:big' under big): the provider's measured row per call class on each card, its named step-aside to the block's flash core inside the provider (LEVER rows= / refused= / plan=, the block's cores=); no kit row table or row floor; AFO_TRIATTN_WORD is a developer override of the word; a shape past the int32 index bound runs the provider default core (index_2p31:<B>x<N>, counted)",
    ),
    "triattn_exact": dict(
        cls="exact", strategy="LOCAL.atlasfold.triattn_exact", module="atlasfold_opt.hooks.triattn_exact", provider="opt_core.kernels.triattn (word exact: the stock cuEquivariance op by name; on cc 8.0 at large N and on a few fp32 cells exact_headsplit — the same library kernel per head, bitwise the stock call)",
        site=f"{TU}:cueq_tri_attn (the cuequiv branch of TriangleAttentionStartingNode / EndingNode.forward; in the fast rows beneath triatt_block, above flash_triattn)",
        expected=("stock_row:", "dtype_", "arch:", "rank", "head_dim_", "single_head", "no_candidate_admitted", "no_calls", "tier_beneath:"),
        what="the stock triangle-attention call through opt_core.kernels.triattn under AFO_TRIATTN_EXACT_WORD (default the tier word exact): the provider's exact cell per (cc, dtype, D, H, N) and stack answers the word exact with the stock op by name (stock_row:cueq), or with exact_headsplit (the same library kernel one head at a time, bitwise the stock call) on cc 8.0 at large N and on a few fp32 cells; no kit size floor; in a fast / big mode it hands every call to the fast tier's binding of the same call beneath it (tier_beneath:<word>)",
    ),
    "triatt_block_exact": dict(
        cls="exact", strategy="LOCAL.atlasfold.triatt_block_exact", module="atlasfold_opt.hooks.triatt_block_exact", provider="opt_core.attn.pair_fused impl=fpf ln=stock core=<the stock cueq_tri_attn call, per call: the stock cuEquivariance op or triattn_exact's exact provider row>",
        site=f"{TU}:TriangleAttentionStartingNode.forward + TriangleAttentionEndingNode.forward (triangle_update.py L314-399 / L425-515; the surround LN -> qkv|g|bias -> [stock attention call] -> gate -> out; in the fast / big rows beneath triatt_block)",
        expected=("backend_torch", "training", "dtype:", "rank", "c:", "below_min_tokens", "above_proven_rows", "cpu", "arch:", "card_rows:", "no-cell:", "below_min_rows", "above_max_rows", "trailing_piece"),
        what="the exact-class construction of the fused triangle-attention surround (prologue and epilogue kernels around the stock attention call): the module's own LayerNorm as installed (stock, ln_bf16's or exactln's), then opt_core.attn.pair_fused.tri_attn_block(ln='stock', x_ln=<it>, residual=False) — ONE prologue kernel x_ln -> q|k|v|g|pair bias with stock rounding, the STOCK attention call (module-global cueq_tri_attn, looked up per call), ONE epilogue kernel sigmoid(g)*o @ W_o — byte-identical to the stock statement; bf16 z at c_z=128, 4x32 heads, N >= AFO_TRIATT_BLOCK_EXACT_MIN_TOKENS (512) and B*N*N <= AFO_TRIATT_BLOCK_EXACT_MAX_ROWS (1280^2, the largest byte-checked call) on cc 9.0; cc 8.0 under the measured row-count rule (hooks/card_rows.py); other cards arch:smNN; cells = the core table's fpf prologue / epilogue rows at (128, 4, 32)",
    ),
    "pair_block_residual": dict(
        cls="fast", strategy="LOCAL.atlasfold.pair_block_residual", module="atlasfold_opt.hooks.pair_block_residual", provider="opt_core.attn.pair_fused impl=fpf residual=True (tri_attn_block epilogue + transition epilogue)",
        site="atlasfold.model.network.block:PairBlock.forward (block.py L240-283: the pair-to-pair step's z += tri_attn_start / tri_attn_end / transition_z adds and z * pair_mask)",
        requires=("triatt_block", "pair_transition"),
        expected=("backend_torch", "training", "dtype:", "rank", "c:", "below_min_tokens", "cpu", "no_calls"),
        what="the residual adds of the two triangle attentions and the pair transition folded into the fused kernels' epilogues (z updated in place; the mask multiply in place); requires triatt_block + pair_transition; N>=512",
    ),
    "pair_transition": dict(
        cls="fast", strategy="LOCAL.atlasfold.pair_transition", module="atlasfold_opt.hooks.pair_transition_fused", provider="opt_core.kernels.transition by TIER word (fast in the fast row, big in the big row: the measured cell's winner per (cc, dtype, c, hidden, n_tokens) — v1 / v2 / af3_fused / the statement; AFO_PAIR_TRANSITION_WORD names another tier or row word, pinned=1)",
        site="atlasfold.model.network.primitives.transition:Transition.forward (outermost: installed after conf_transition_chunk / pair_transition_chunk / transition_exact, whose wrapper is this lever's stock callable)",
        expected=("training", "dtype:", "rank", "c:", "factor:", "cpu", "stock_row:", "refused:"),
        what="opt_core.kernels.transition.transition(x, W, word=<tier>, residual=False): ONE fused kernel LN -> W_a|W_b -> silu(a)*b -> W_out chosen by the provider's cells (the 4*c_z hidden never reaches HBM; weights packed once per module by the provider); bf16 pair transitions at c_z=128, factor 4 — the LM stack's, the Pairformer's and, under the runners' bf16 autocast, the confidence pair stack's — at every size the provider's cells serve a kernel row; every declined call (fp32 inputs: the diffusion PairConditioning transitions, C=384 / 768 single track, factor-2 diffusion transitions, a cell naming the statement `stock_row:<cell>`, a provider refusal `refused:<kind>`) runs the statement beneath, counted; LEVER fields word= rows=<row:n> plan=<C/N:row/class> pinned= copies=",
    ),
    "lm_sdpa": dict(
        cls="fast", strategy="LOCAL.atlasfold.lm_sdpa",
        site="atlaslm.layers.attention:MultiHeadAttention.forward (L97-105: eager logits kept for export; softmax*V via F.scaled_dot_product_attention)",
        expected=("no_logits_requested",),
        what="keeps the bf16 QK^T logits AtlasFold consumes (model.py L460-466) and replaces the eager softmax+matmul by SDPA with the block-diagonal seq_id mask",
    ),
    "transition_exact": dict(
        cls="exact", strategy="LOCAL.fused_transition", module="atlasfold_opt.hooks.transition_exact", provider="opt_core.kernels.transition by TIER word (select(word) per (cc, dtype, C, hidden, n_tokens, running stack); fed the module's OWN LayerNorm output; residual=False)",
        site="atlasfold.model.network.primitives.transition:Transition.forward (every SwiGLU Transition: LM stack c384, Pairformer pair c128 + single c768, confidence pair stack c128, diffusion c768 fp32; outer of conf_transition_chunk in exact, beneath pair_transition in fast / big — big's pair_transition_chunk wrapper hands its row blocks down to it)",
        expected=("training", "dtype:", "rank", "cpu", "stock_row:", "refused:", "above_proven_rows", "no_calls"),   # refused:<kind> = ANY provider Refusal (no cell, no bitwise record on this stack, below the recorded floor, ...): the module's own statement runs, by name; above_proven_rows = the word of a kit-side size ceiling (accepted; transition_exact installs none)
        what="the SwiGLU Transition bound BY TIER WORD to the core's transition provider: AFO_TRANSITION_EXACT_WORD (default `exact`: the provider's exact-class cell winner per shape — a carried row (v1 fed the module's own LayerNorm output at bf16 pair c128) only where the provider's table records it BITWISE against the statement on the running stack at or below the call's size, the STOCK statement everywhere else (the single track's c384 / c768 cells, pair cells whose exact winner is the statement, no vouch on this stack), by name; a row word pins that row where the provider admits it, printed pinned=1; tolerance rows / tier words fast|big are not exact-class -> the lever installs skipped by name) — ONE kernel W_out(silu(x_ln W_a^T) * (x_ln W_b^T)), the 4c hidden never in HBM, byte-identical to the stock statement; no kit-side size floor, ceiling or row pin (the provider's vouch records decide); LEVER fields word= plan=<C/N:row/class or stock or refused:kind> rows= pinned=",
    ),
    "ln_bf16": dict(
        cls="exact", strategy="LOCAL.atlasfold.ln_bf16", module="atlasfold_opt.hooks.ln_bf16",
        site="atlasfold.model.network.primitives.normalization:LayerNorm.forward (L31-43)",
        expected=("fp32_input", "fp32_params", "cpu"),
        what="LayerNorm of a bf16 input on a bf16-parameter module (lm_stack / main_stack) as ONE F.layer_norm call with autocast off (fp32 statistics in-kernel, one rounding to bf16) instead of x.float() -> fp32 LayerNorm -> .to(bf16); bit-identical on torch 2.7.1+cu128; fp32 inputs / fp32-parameter modules take the stock statement",
    ),
    "diffusion_bf16": dict(
        cls="fast", strategy="LOCAL.atlasfold.diffusion_bf16",
        site="atlasfold.model.network.diffusion_head:DiffusionHead.sample (model.py L316-319 disables autocast around it)",
        expected=(),
        what="bf16 autocast over the 12-block diffusion token transformer; the atom encoder/decoder keep their own fp32 islands (diffusion_head.py L151/L178) and precision=32 linears",
    ),
    "atom_sdpa": dict(
        cls="fast", strategy="LOCAL.atlasfold.atom_sdpa", module="atlasfold_opt.hooks.atom_sdpa",
        site="atlasfold.model.network.attention:Attention.forward (L42-100; the rank-5 [B,N,W,L,c] calls = diffusion_transformer.py AtomTransformerBlock.attention (CrossAttention) of the AtomEncoder / AtomDecoder stacks, 3 blocks x 2 stacks x steps x sample chunks); installed after dit_sdpa on the same attribute (composes over it)",
        expected=("disabled", "rank", "high_precision", "kv_lead", "bias_form"),
        what="the windowed atom attention (56 query x 168 key atoms per window, 2 heads x 48, fp32) through torch's fused SDPA kernel: q/k/v viewed [B,N,W,H,L,D] without a copy, the stock mask+pair bias summed once into a 16-aligned buffer, one fused call per (b, n) over the W windows (mem-efficient: fp32 in/out, 3xTF32 products, online softmax) instead of the MATH backend's contiguous copies + materialised [B,N,W,H,56,168] logits + safe-softmax passes + two bmm; rank-3/4 and use_high_precision calls keep the statement below by name; AFO_ATOM_SDPA=0 -> every call `disabled`",
    ),
    "atom_tf32": dict(
        cls="fast", strategy="LOCAL.atlasfold.atom_tf32", module="atlasfold_opt.hooks.atom_tf32",
        site="atlasfold.model.network.atom_attention:AtomAttentionStack.forward (L44-100; called by AtomEncoder.forward L128 and AtomDecoder.forward L196 inside DiffusionModule.forward's two autocast-disabled islands, diffusion_head.py L151/L178) + rel_pos_encoding:AtomRelativePositionEncoding.forward (KEEP_COORD guard)",
        expected=("disabled", "cpu", "cc<8"),
        what="TF32 tensor-core GEMMs (torch.backends.cuda.matmul.allow_tf32 True on entry, the entry value restored on exit) for the fp32 windowed atom transformers of the denoiser: q/k/v/gate/out + AdaLN + SwiGLU linears, linear_pair_bais and the MATH-SDPA bmm pair; the coordinate-facing precision=32 linears (AtomEncoder.linear_in, AtomDecoder.linear_out), random_augmentation and the relpos producer stay IEEE fp32 (outside the call / guarded); host state at warm-up + capture, so denoiser_graph records and replays the TF32 kernels; AFO_ATOM_TF32=0 -> every call `disabled`",
    ),
    "denoiser_graph": dict(
        cls="fast", strategy="LOCAL.atlasfold.denoiser_graph", module="atlasfold_opt.hooks.denoiser_graph",
        site="atlasfold.model.network.diffusion_head:DiffusionModule.forward (L122-181) + DiffusionHead.sample (L283-373: one roll-out = one graph lifetime)",
        expected=("above_gate", "cpu"),
        what="the denoiser network (atom encoder -> 12 DiT blocks -> atom decoder, ~600 launches per call) captured into a torch.cuda.CUDAGraph at the first call of each "
             "sample() roll-out and replayed at every step (r_noisy/single_cond copied into static buffers, batch + pair_bias held by reference, output cloned out of the pool); "
             "tokens > AFO_DENOISER_GRAPH_MAX_TOKENS (1024) -> the eager statement, counted above_gate; capture_failed:<Exc> / recapture_storm -> eager for the roll-out, gate refused. "
             "Levers whose Python body runs inside the captured call (dit_sdpa, ln_bf16) count only warm-up + capture + eager calls per roll-out while their kernels replay every step",
    ),
    "graph_reuse": dict(
        cls="fast", strategy="LOCAL.atlasfold.graph_reuse", module="atlasfold_opt.hooks.graph_reuse", requires=("denoiser_graph",),
        site="atlasfold_opt.hooks.denoiser_graph:RollOut (its adopt / before_capture / after_capture / owns hook points, through "
             "ROLLOUT_FACTORY) + atlasfold.model.model(_multimer):AtlasFold(_Multimer).inference (the pre-trunk drop of a kept set of another (B, L))",
        expected=("first", "key", "disabled"),
        what="denoiser_graph's captured graph(s) kept ACROSS items: graph + private pool + static buffers + every tensor the captured kernels read by "
             "reference (batch dict, pair_bias, sampler_hoist's roll-out leaves: one ordered structural walk, RefTable) stay alive after sample() returns; "
             "the next roll-out of the same structure (bucket, batch, sample chunk, dtype, autocast: denoiser_graph's key without addresses) runs the "
             "eager warm-up call, copies its walk's storages into the kept ones, r_noisy / single_cond into the static inputs and REPLAYS — no capture "
             "(0.3-0.6 s per item at the 512 bucket); the first replay of an adopted graph must equal the warm-up output (else `probe`: dropped, captured, "
             "gate refused). One item's graphs alive at a time: a capture of another key, an item of another (B, L) before its trunk, an item above the "
             "token gate drop the kept set first (peak memory of every item as without the lever); AFO_GRAPH_REUSE=0 = installed, inert (`disabled`)",
    ),
    "pae_stream":        dict(cls="exact", strategy="LOCAL.atlasfold.pae_stream", module="atlasfold_opt.hooks.pae",
                             site="atlasfold.model.network.confidence_head:ConfidenceHead_Monomer.forward + atlasfold.model.utils.confidence_metrics:compute_pae/compute_ptm"),
    "pae_stream_m":      dict(cls="exact", strategy="LOCAL.atlasfold.pae_stream_m", module="atlasfold_opt.hooks.pae_multimer",
                             site="atlasfold.model.network.confidence_head:ConfidenceHead_Multimer.forward + atlasfold.model.utils.confidence_metrics:compute_pde/compute_*_from_probs",
                             what="AtlasFold-M: per-sample PAE/PDE reduction with the stock functions instead of two fp32 [B,N,L,L,64] stacks + their softmax (monomer runs: n/a)"),
    "conf_transition_chunk": dict(cls="exact", strategy="LOCAL.atlasfold.conf_transition_chunk", module="atlasfold_opt.hooks.transition_chunk",
                             site="atlasfold.model.network.primitives.transition:Transition.forward (only under ConfidenceHead_{Monomer,Multimer}.forward)",
                             what="the confidence heads' fp32 pair Transition applied in row blocks (SwiGLU intermediate <= AFO_TRANSITION_CHUNK_MIB per block) instead of one [B,L,L,4c_z] fp32 tensor"),
    "pair_transition_chunk": dict(cls="exact", strategy="LOCAL.atlasfold.pair_transition_chunk", module="atlasfold_opt.hooks.transition_chunk",
                             site="atlasfold.model.network.primitives.transition:Transition.forward outside the confidence heads (LMStack + Pairformer blocks block.py L281, diffusion PairConditioning transitions diffusion_transformer.py L137)",
                             what="every pair-shaped Transition of the trunk and the diffusion pair conditioning applied in row blocks (SwiGLU intermediate <= AFO_TRANSITION_CHUNK_MIB per block) instead of one [B,L,L,8c_z] bf16 / fp32 tensor (32+16+16 GiB at L=4,096); AFO_PAIR_TRANSITION_CHUNK=0 = off; a pair call a transition kernel lever served first (transition_exact / pair_transition install outside this wrapper) counts as the named fallback upstream_transition_exact / upstream_pair_transition"),
    "distogram_offload": dict(cls="exact", strategy="LOCAL.atlasfold.distogram_offload", module="atlasfold_opt.hooks.distogram",
                             site="atlasfold.model.network.distogram_head:DistogramHead.forward"),
    "relpos_lazy":       dict(cls="exact", strategy="LOCAL.atlasfold.relpos_lazy", module="atlasfold_opt.hooks.relpos",
                             site="atlasfold.model.model:AtlasFold.compute_rel_pos_encoding (+multimer) -> LinearNoBias.forward / PairConditioning.forward / AtomAttentionStack.forward"),
    "sampler_hostsync":  dict(cls="exact", strategy="LOCAL.atlasfold.sampler_hostsync", module="atlasfold_opt.hooks.sampler_hostsync",
                             site="atlasfold.model.network.diffusion_head:DiffusionHead.sample (re-stated: innermost patch of sample, installed before every lever that wraps it) "
                                  "<- utils/geometry/random_augment.py do_centering L91 / get_center L47 (mask.any()), diffusion_head.py L343 (torch.tensor(c_noise) upload)",
                             expected=("disabled",),
                             what="the roll-out's per-step host syncs removed: do_centering's and get_center's `mask.any()` on the roll-out-constant atom mask evaluated ONCE per "
                                  "roll-out and answered from that bool, and the per-step `torch.tensor(self.c_noise(t_hat), device=...)` scalar upload replaced by ONE fp32 table "
                                  "of the whole schedule built from the loop's own float expressions (same double->float32 rounding per element), each step reading its [1,1] row "
                                  "by a device view; every other statement, the RNG draws and their order verbatim -> bit-identical; host syncs inside sample() 3/step (600/roll-out) -> 2/roll-out. "
                                  "AFO_SAMPLER_HOSTSYNC=0 = installed, stock body every roll-out (counted disabled); MODEL_OPT_LEVERS_OFF=sampler_hostsync removes it from the row; "
                                  "refuses by name when sample() is already wrapped at install (not_innermost) or a re-stated stock text's digest changed (source:<fn>)"),
    "sampler_hoist":     dict(cls="exact", strategy="LOCAL.atlasfold.sampler_hoist", module="atlasfold_opt.hooks.sampler_hoist",
                             site="atlasfold.model.network.diffusion_head:DiffusionHead.sample (one roll-out = one state) -> PairConditioning.forward / SingleConditioning.forward "
                                  "(diffusion_transformer.py) / AtomEncoder.forward + AtomAttentionStack.forward (atom_attention.py) / AtomTransformerStack.forward + the atom blocks' "
                                  "AdaLN / gate sub-modules (per instance); consumed at attention.py L94-98",
                             expected=("disabled", "atom_bias:atom_sdpa"),
                             what="the sampler's step-invariant statements evaluated ONCE per DiffusionHead.sample roll-out by the same ops on the same operands in the same dtype and "
                                  "re-used by reference at every step (bit-identical by construction; the loop itself is not re-stated): the 12 DiT blocks' additive bias "
                                  "mask_bias + pair_bias[i] folded in place into the pair-bias tensor (no extra memory), the atom encoder's c = embed(aatype) and masks, both atom "
                                  "stacks' LocalAttentionIndex + linear_pair_bais(atom_rel_pos) (the relpos_lazy producer then replays twice per roll-out instead of twice per "
                                  "step), the atom-level attention mask and conditioning windows, the atom blocks' AdaLN conditioning branches and gates (instance memo of the "
                                  "leaves on held tensors), their additive biases, SingleConditioning's proj_single_cond(cat(s, aatype)). Composes with denoiser_graph (held "
                                  "tensors are created at the warm-up call and read by reference by the graph), relpos_lazy, dit_sdpa, diffusion_bf16, sampler_hostsync. "
                                  "AFO_SAMPLER_HOISTS=<list|-name> ablates per hoist; AFO_SAMPLER_HOIST=0 = installed, every roll-out stock (counted disabled); "
                                  "MODEL_OPT_LEVERS_OFF=sampler_hoist removes it from the row; a head in training / under grad -> stock (train); a stock function whose source "
                                  "digest changed -> that hoist off at install (off=<hoist>:source)"),
    "exactln": dict(
        cls="exact", strategy="LOCAL.atlasfold.exactln", module="atlasfold_opt.hooks.exactln", provider="opt_core.kernels.ln by TIER word (exact -> the carried bit-for-bit ATen replica exactln where the provider records it bitwise on the running stack, else ATen by name; fast / big -> the measured cell's tolerance-class winner: fastln / ln_rows / exactln / ... or ATen) per call class (cc, form, C, rows, alignment, capture); AFO_EXACTLN_WORD names another tier or row word (pinned=1)",
        site="atlasfold.model.network.primitives.normalization:LayerNorm.forward (L31-43), every call: the trunk's triangle-attention / pair-transition / pair-to-single LayerNorms, PairConditioning, the confidence pair stack, the single track, the DiT token rows, the atom rows; installed after ln_bf16 on the same attribute (a call the provider resolves to a stock row runs the statement below, by name)",
        expected=("disabled", "cpu", "dtype:", "param_form", "normalized_shape", "stock_row:", "no_affine:", "refused:", "no_core:"),
        what="the stock statement F.layer_norm(x.float(), (C,), w.float(), b.float(), eps).to(x.dtype) in its three forms (fp32 x / fp32 parameters; bf16 x / bf16 parameters; bf16 x / fp32 parameters with the bf16 store) resolved by the provider's cells per call class and served by the resolved row; a STOCK row (aten / aten_autocast), AFO_EXACTLN=0 (`disabled`), CPU tensors, fp16 inputs, mixed parameter dtypes, a multi-axis norm, a no-affine norm resolved to a row this kit has not run without gamma / beta (`no_affine:<row>`), a provider refusal (`refused:<kind>`) take the statement below BY NAME; LEVER fields word= pinned= rows=<row:n> plan=<form/C/cell:row>",
    ),
    "atom_kdedup":       dict(cls="exact", strategy="LOCAL.atlasfold.atom_kdedup", module="atlasfold_opt.hooks.atom_kdedup",
                             site="atlasfold.model.network.attention:CrossAttention.forward (L234-294: the atom blocks' conditioned windowed cross attention, "
                                  "diffusion_transformer.py AtomTransformerBlock L354; key side L269 = normalization.py AdaLN.forward L71-76 + Attention.forward L69 "
                                  "linear_k / linear_v) + the atom Attention modules' linear_k / linear_v instances (k / v hand-over by identity for the one L285 call)",
                             expected=("disabled", "below_exact_min_rows"),
                             what="the windowed atom attention's KEY side evaluated on the atom ROWS instead of on the 3x-overlapping key windows: a_k's strides prove it is "
                                  "rows.unfold (LocalAttentionIndex.to_k: window 12 residues, step 4, over one zero-padded copy), and LayerNorm, the AdaLN combine and the "
                                  "two projections are row-wise maps, so adaln_a_k and linear_k / linear_v run once per row of that copy ([B,N,(L+8)*14,c], ~1/3 of "
                                  "[B,N,W,168,c]) with the same modules / statements / dtypes and k / v go back to Attention.forward as the SAME overlapping window views "
                                  "(as_strided with a_k's stride pattern) — the MATH SDPA (exact) and atom_sdpa's fused kernel (fast) read element-identical operands; "
                                  "the conditioning leaves join sampler_hoist's roll-out memo when it holds single_cond_k (else per call on the rows); nothing held past "
                                  "a call, row-shaped intermediates replace window-shaped ones (peak can only fall); class by measurement: exact (byte-identical vs off "
                                  "under --det 1) -> exact + fast rows; AFO_ATOM_KDEDUP=0 = installed, every call the statement below (disabled); a key / cond tensor that "
                                  "is not an unfold view (kv_layout / cond_layout) or a rank other than 5 takes the statement below by name"),
    "dit_apb": dict(
        cls="fast", strategy="LOCAL.atlasfold.dit_apb", module="atlasfold_opt.hooks.dit_apb", provider="opt_core.kernels.apb by the MODE'S TIER WORD (fast | big | exact): select(cc, dtype, dit_h16d48, n_tokens, samples, eager|graph) names the row per call class from the measured cell table (APB_CELLS); no kit row / card / size table",
        site="atlasfold.model.network.attention:Attention.forward (L42-100), the diffusion transformer's rank-4 [B,N,L,c] calls (12 DiT blocks x steps per roll-out); installed after sampler_hoist, atom_sdpa and atom_kdedup on the same attribute (composes over them: rank-3 / rank-5 calls reach the wrappers below unchanged); in the exact row too (the exact word's door: serves nothing today, see what)",
        expected=("disabled", "rank", "high_precision", "kv_lead", "bias_form", "cpu", "no_core", "stock_row:", "refused:"),
        what="q/k/v from the module's linears viewed [B,N,H,L,D], the stock finite mask bias + pair_bias.to(q.dtype) (the statement sampler_hoist folds once per roll-out) handed per batch element as ONE [1,H,L,L] bias shared by the N samples -> opt_core.kernels.apb.pair_bias_attention under the mode's tier word: fast / big serve the cell's measured row (fpf_apb / apb_attn / l3a / sdpa:* ... per cc x dtype x N-bucket x eager|graph; the timing form follows the call so denoiser_graph's warm-up compiles the row the capture records; a row refusing at launch is followed once to the provider's named fallback row, refused=<row>:<kind>-><row> on the line; tolerance class, deterministic, capturable); exact serves a row only where the table vouches an exact-class row bitwise ON THIS STACK against this engine's statement (torch SDPA MATH backend, the rank-5 stock call's route) -- the provider's exact class replicates the fp32 memory-efficient statement (sdpa_upcast) and carries no vouch on torch 2.7.1+cu128, so every exact call class takes the module's own statement BY NAME (stock_row:<arm>, decided by pure selection, no launch; census NAMED_FALLBACK word=exact) and bytes never change; plan=<N><e|g>:<row> / rows=<row>:<n> / refused= on the LEVER line, the core's cell census at exit; steps aside by name for rank-3/5 calls (rank), use_high_precision, cross-attention leading dims (kv_lead), a bias without per-row/per-head values or a per-sample mask (bias_form), CPU tensors, no_core; AFO_DIT_APB_WORD=<row[:variant]> = a developer override naming one provider row for every call in fast / big (0|off = installed, every call disabled; under --mode exact only the exact word is a word: anything else installs skipped not_an_exact_word:<w>)",
    ),
    "atom_bf16":         dict(cls="fast", strategy="LOCAL.atlasfold.atom_bf16", module="atlasfold_opt.hooks.atom_bf16",
                             site="atlasfold.model.network.atom_attention:AtomAttentionStack.forward (L44-100; called from AtomEncoder.forward L169 and AtomDecoder.forward "
                                  "L202 inside DiffusionModule.forward's autocast(enabled=False) islands, diffusion_head.py L151 / L178) + the KEEP_COORD guard on "
                                  "atlasfold.model.network.rel_pos_encoding:AtomRelativePositionEncoding.forward",
                             expected=("disabled", "cpu"),
                             what="torch.autocast(bf16) around the fp32 windowed atom transformers of the denoiser (3 blocks x 2 stacks per call: q/k/v/gate/out, "
                                  "AdaLN, SwiGLU and linear_pair_bais GEMMs in bf16 with fp32 accumulation; LayerNorm statistics and the residual stream stay fp32), "
                                  "the result returned in the caller's dtype; the coordinate-facing precision=32 linears (AtomEncoder.linear_in, AtomDecoder.linear_out), "
                                  "random_augmentation and the roll-out arithmetic are outside the call and the relpos producer is guarded (autocast off) — KEEP_COORD; "
                                  "host state at warm-up + capture so denoiser_graph replays the bf16 kernels; tolerance class (the identity band); the outermost "
                                  "AtomAttentionStack.forward wrapper (over atom_tf32 / sampler_hoist's prologue / relpos_lazy); AFO_ATOM_BF16=0 = every call disabled"),
    "atom_rows":         dict(cls="fast", strategy="LOCAL.atlasfold.atom_rows", module="atlasfold_opt.hooks.atom_rows",
                             site="atlasfold.model.network.diffusion_transformer:ConditionedTransitionBlock.forward (L54-59; the atom blocks' conditioned transition, "
                                  "AtomTransformerBlock L359: rank-5 [B,N,W,56,96] rows; the token transformer's rank-4 calls of the same class keep the statement below by name)",
                             expected=("disabled", "cpu", "rank"),
                             what="the atom blocks' conditioned transition on opt_core.kernels.dtk_kernels row kernels: ln_modulate (LayerNorm + the AdaLN combine, the "
                                  "conditioning factors as per-atom rows serving the N sample rows by period — never expanded), the stock SwiGLU GEMM then swiglu "
                                  "(silu(x1)*x2 in one pass), the stock output GEMM then gate_residual (the sigmoid gate in one pass): 8 elementwise / LayerNorm "
                                  "kernels per block -> 3, fp32 statistics, one rounding to the output dtype (the autocast dtype under atom_bf16); cond factors join "
                                  "sampler_hoist's roll-out memo when it holds cond; tolerance class (identity band), deterministic; AFO_ATOM_ROWS=0 = every call disabled"),
    "lever_report": dict(
        cls="exact", strategy="LOCAL.atlasfold.lever_report", site="atexit", expected=(),
        what="one LEVER line per lever + the EXIT tally at interpreter exit (opt_core.report.register_exit_tally)",
    ),
    "output_overlap":    dict(cls="exact", strategy="LOCAL.atlasfold.output_overlap", module="atlasfold_opt.hooks.output_overlap",
                             site="atlasfold.runner_multimer:MultimerFoldingRunner.fold_iter_batch (runner_multimer.py L241-325, re-stated: forwards on the caller's thread, "
                                  "the batch's host statements in ONE spawn-context worker process; digest-guarded) + ProteinMultimerOutput.to_mmcif/to_pdb (the writer's "
                                  "text served from the worker's precompute) + atlasfold.runner:FoldingRunner.fold_iter_batch (stock by name: runner:monomer)",
                             expected=("disabled", "runner:monomer"),
                             what="the multimer batch generator pipelined WITHOUT a thread: the stock featurize + model_run statements stay on the caller's (main) thread "
                                  "in the stock batch/seed order; each batch's host arrays go (pickled on that thread, between forwards) to ONE multiprocessing "
                                  "spawn-context worker process (started at activation behind the model load; CUDA_VISIBLE_DEVICES='' and no kit word: a stock, "
                                  "device-less host interpreter, nice +10) that runs the stock _make_outputs (5x numba SASA + Protein objects + confidence dicts), "
                                  "the stock ranking + MultimerFoldingOutput statements and the CLI writer's per-sample to_mmcif|to_pdb text + confidence_scores; "
                                  "finished batches are collected on the caller's thread in the stock order and yielded (batch k after forward k+1), so batch k's "
                                  "SASA / objects / text are computed while forward k+1 runs and the writer's statements cost milliseconds; the run's LAST batch is the "
                                  "caller's at once (nothing follows it: a single-record run is the stock body + an idle worker) and a hand-over waits for the worker's "
                                  "boot (never beside a forward); value statements, operands "
                                  "and order unchanged -> byte-identical files (the equality leg); device peak unchanged, host: <= 3 batches of arrays + the worker "
                                  "(~1 GiB RSS); a worker error is re-raised in the caller with its traceback, a dead worker's batch is recomputed on the caller BY "
                                  "NAME (gate refused at exit either way - never silent); AFO_OUTPUT_OVERLAP=0 = the stock generator body (counted disabled, no worker); "
                                  "LEVER line served=<batches> worker=process threads=0 queued= last= inline= wait_s= boot_wait_s= post_s= overlapped_s= xfer_s= "
                                  "max_inflight= memo_served= worker_boot_s= worker_rss_mib= worker_pss_mib= worker_uss_mib="),
}


def describe(name: str) -> dict:
    if name not in LEVERS:
        raise KeyError(f"unknown lever {name!r}; known: {sorted(LEVERS)}")
    return dict(LEVERS[name], name=name)
