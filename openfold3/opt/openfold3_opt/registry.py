"""Lever registry: metadata for every optimization the modes compose, keyed by the switches the add-ons' own hooks read.

No values live here: the switches and their values are the mode lines in modes.py (quoted from the kit READMEs); each entry names the
kit file and line the lever was read from, the switch it answers to, its numerics class, the stderr line the kit prints when it is
installed and the stock module whose first import makes the kit's finder install it (`target`: the moment the lever's own record —
`stack.lever_applied()` reads the add-ons' flags, state dicts and installed lists — can say applied or unavailable; before that import the
lever is pending). `tests/test_modes.py` covers the mode table `registry.LEVERS` is keyed against; `tests/test_kit_paths.py` locks
every `source` here against a file that exists in the kit tree.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

FORWARD = "forward"
OUTPUT = "output"                                                   # the phase after the model forward: the engine's output writer (a lever here moves the item's wall time, not the model forward)
EXACT, TOLERANCE = "exact", "tolerance"
T_LINEAR = "openfold3.core.model.primitives.linear"                 # fast_inference/of3_levers/sitecustomize.py
T_DIFFUSION = "openfold3.core.model.structure.diffusion_module"     # sitecustomize.py
T_MODEL = "openfold3.projects.of3_all_atom.model"                   # trunk_kernels/of3t_hook/sitecustomize.py:60
T_WRITER = "openfold3.core.runners.writer"                       # opt/openfold3_opt/cells/fastjson.py (installed at activation; patched when the engine imports its writer)
T_HEADS = "openfold3.core.model.heads.prediction_heads"          # opt/openfold3_opt/hooks/confhead/sitecustomize.py (_TARGET)
T_CKPT = "openfold3.core.utils.checkpoint_loading_utils"        # opt/openfold3_opt/cells/ckpt_mmap.py (installed at activation; patched when the runner imports its checkpoint reader)
T_MSA_IO = "openfold3.core.data.io.sequence.msa"               # opt/openfold3_opt/cells/hostfeat.py (installed at activation; patched when the engine imports its MSA reader, before the DataLoader forks)


@dataclass(frozen=True)
class Lever:
    name: str
    kit: str                                   # the add-on table's key (modes), or modes.PACKAGE_HOOKS key for a lever the package itself ships (confhead)
    cls: str                                   # forward
    tier: str                                  # exact | tolerance
    env_keys: Tuple[str, ...]                  # the switch(es) the kit reads for it (names only; values in modes.LINES)
    description: str
    marker: str                                # the stderr line (prefix) the kit prints when the lever is installed; "" when it prints none
    source: str                                # where the switch was read (kit file:line)
    modes: Tuple[str, ...] = ()                # informational: the house modes / lines that carry it, spelled `mode` or `mode/line` of modes.LINES (exact, fast, big/resident, big/tp)
    target: str = ""                           # the stock module whose import triggers the kit's install of this lever ("" = not applied by a hook)


LEVERS: Dict[str, Lever] = {l.name: l for l in [
    Lever("fast_init", "fast_inference", FORWARD, EXACT, ("OF3_FAST_INIT",),
           "U0: skips the CPU-side random weight initialisation the checkpoint load overwrites (model construction 68 s -> 14 s)",
           "[of3_levers] fast-init enabled", "fast_inference/README.md; of3_levers/sitecustomize.py", ("exact", "fast", "big/resident"), T_LINEAR),
    Lever("cuda_graphs", "fast_inference", FORWARD, EXACT, ("OF3_CUDA_GRAPHS",),
           "C3: captures one denoising step of the diffusion sampler per input shape and replays it for the other steps and for later items whose step reads the same shapes (OPENFOLD3_OPT_GRAPHS_KEEP; census kept=); memoised, per-item checked gather tables",
           "[of3_graphs]", "fast_inference/README.md; of3_levers/sitecustomize.py; of3_levers/of3_graphs.py", ("exact", "fast"), T_DIFFUSION),
    Lever("graphs_strict", "fast_inference", FORWARD, EXACT, ("OF3_GRAPHS_STRICT",),
           "capture failure raises instead of the eager fallback (the graphed arm is proven graphed: `invariants:` reads fallbacks=0)",
           "", "of3_levers/of3_graphs.py:276", ("exact", "fast"), T_DIFFUSION),
    Lever("templ_distinct", "trunk_kernels", FORWARD, EXACT, ("OF3T_TEMPL_DISTINCT",),
           "with --use-templates false the 4 identical dummy templates run through the template pair stack once, re-expanded before the stock reduction",
           "[of3t_levers] installed:", "trunk_kernels/README.md; of3t_hook/sitecustomize.py:31,79-80", ("exact", "fast", "big/resident"), T_MODEL),
    Lever("paircache", "trunk_kernels", FORWARD, EXACT, ("OF3T_PAIRCACHE",),
           "D1: the step-invariant conditioned pair representation and the 24 per-block pair biases of the token diffusion transformer computed once per rollout, not 200x; the biases held head-major ([*, H, N, N] with 16-byte aligned rows) so the attention levers read them in place",
           "[of3t_paircache] installed", "trunk_kernels/README.md; of3t_hook/sitecustomize.py:32,81-82; of3t_hook/of3t_paircache.py", ("exact", "fast"), T_MODEL),
    Lever("trimul_cueq", "trunk_kernels", FORWARD, TOLERANCE, ("OF3T_TRIMUL",),
           "triangle multiplicative update through cuEquivariance's fused kernel in the fp32/TF32 precision OpenFold3 runs (rounding points change)",
           "[of3t_levers] installed:", "trunk_kernels/README.md; of3t_hook/of3t_levers.py", (), T_MODEL),                # on no line: the pair cells serve the triangle multiplication through the core's provider; switch it on with OF3T_TRIMUL=cueq
    Lever("trimul_v4", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_PAIR",),
           "the pair stack's two TriMul statements on the core's fused TriMul cell (opt_core.kernels fpf_trimul_v4: LayerNorm + gated dual projection + contraction + LayerNorm-out + gate/projection + residual per direction; c_z = c_hidden in {128,256}, N in the cell's range) — class-wide on PairBlock (trunk, MSA-module, confidence stacks); the template stack (c = 64) is the provider's (trimul_provider) and out-of-range N run the line's own TriMul TriMul by name (pairfused census fallback:<reason>)",
           "[openfold3-opt/pairfused] installed", "opt/openfold3_opt/cells/pairfused.py; opt/openfold3_opt/hooks/cells/sitecustomize.py", ("fast", "big/resident"), T_MODEL),
    Lever("trimul_provider", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_TRIMUL_PROVIDER",),
           "the TriMul cell's row per call class (GPU, precision, c_z, c_hidden, token bucket, direction) chosen by the core's ONE triangle-multiplication provider (opt_core.kernels.trimul: every carried implementation a named row — v4 / tmk3_fast / tmk3_exact / bz2 / tx_sm90a / esm_v5_fwd / esm_v61 / cueq … — and the measured cell table TRIMUL_CELLS.json), word fast in this kit's measured row order (the line's tier word for every class): H100 — the kit row `v4` (this cell's own fpf_trimul_v4 statement, byte for byte) at the pairformer / MSA-module / confidence width (128,128) at every size (the cell's fast row on this stack; the ESM engine's route wins the core's race there only with a c-128 launch cell the provider does not bind yet), the provider's `tmk3_fast` (then `tmk3_exact`) at the template pair stack's (64,64), which the kit row does not serve; A100 — the kit row at (128,128), the template width on the line. A row the provider refuses for a class is counted by its word and the next asked; the caller's OPENFOLD3_OPT_TRIMUL_WORD names any provider word for an ablation; left off (MODEL_OPT_LEVERS_OFF=trimul_provider) the cell serves v4 / the line as before. Census `LEVER name=trimul_provider word=… rows=<row>:<n> kit=v4:<n> line=… skip=… witness=… cells=…`; tolerance class (the first statement per class and row is witnessed against the class's reference, a non-finite or gross one refused by name)",
           "[openfold3-opt/pairfused] trimul_provider:", "opt/openfold3_opt/cells/pairfused.py; opt/openfold3_opt/of3_trimul.py", ("fast", "big/resident"), T_MODEL),
    Lever("triatt_block", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_PAIR", "OPENFOLD3_OPT_PAIR_IMPL"),
           "the pair stack's starting + ending triangle attention as the core's fused block (opt_core.attn.pair_fused.tri_attn_block: LayerNorm -> q|k|v|g + pair-bias prologue, the core's flash triangle attention, sigmoid-gate * o @ W_o epilogue adding its output into z in place, a transposed scatter for the ending node, which reads z^T by address math); impl fpf (OPENFOLD3_OPT_PAIR_IMPL); fp32 stream = the module's own LayerNorm feeding bf16 kernels, bf16 stream = fully fused; shapes the core's cell table does not attest run the line's flash dispatcher by name",
           "[openfold3-opt/pairfused] installed", "opt/openfold3_opt/cells/pairfused.py", ("fast", "big/resident"), T_MODEL),
    Lever("triatt_provider", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_PAIR_CORE",),
           "the fused triangle-attention block's attention core chosen per call class (GPU, dtype, head dim, heads, key count, call form) by the core's ONE triangle-attention provider (opt_core.kernels.triattn, cell table TRIATTN_CELLS.json) on the line's TIER word `fast` — the cell's measured fast-class row per class on each card (H100: the sealed sm_90a extension / the sealed triangle-attention package at head dim 32, the package or k2b at head dim 16; A100: the package's sm_80 member; k2b / k2 where a package has no prebuilt for the torch ABI), no kit-side row order; a row that cannot serve is refused by name and the cell's named fallback serves, counted; left off, the block's flash core (the block before this lever)",
           "[openfold3-opt/pairfused] installed", "opt/openfold3_opt/cells/pairfused.py; opt/openfold3_opt/of3_triattn.py", ("fast", "big/resident"), T_MODEL),
    Lever("pair_transition", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_PAIR",),
           "the trunk's SwiGLU transitions as the core's fused transition (opt_core.attn.pair_fused.transition: LayerNorm -> W_a|W_b -> silu(a)*b -> W_out, the n*c hidden never in HBM): PairBlock / MSABlock transition instances (pairformer, confidence, MSA-module pair, template pair; the MSA transition); where the core's fpf_transition_v2 kernel serves, the row mask and the residual add are folded and PairBlock's tail is one kernel writing z + mask*T(z) in place",
           "[openfold3-opt/pairfused] installed", "opt/openfold3_opt/cells/pairfused.py", ("fast", "big/resident"), T_MODEL),
    Lever("rollout_bf16", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_ROLLOUT",),
           "the diffusion denoiser step — DiffusionModule (conditioning, the 24-block diffusion transformer, the token-level projections), 200 steps over the sample batch, and the graphed step's eager per-rollout work (the pair cache's token-pair buffers) — under torch.autocast(cuda, bfloat16) instead of upstream's fp32 rollout, with DiffusionModule.atom_attn_enc / .atom_attn_dec (the modules that read and write atom coordinates) under torch.autocast(cuda, float32) and the sampler's rotation / noise / update arithmetic left in upstream's fp32 (bf16's 8-bit mantissa on a coordinate is 0.06-0.25 A of bond-length noise): matmul-class ops bf16 with fp32 accumulation, LayerNorm / softmax statistics fp32 (autocast's op lists); holds inside the CUDA-graphed step",
           "[openfold3-opt/rollout] installed", "opt/openfold3_opt/cells/rollout.py", ("fast", "big/resident"), T_MODEL),
    Lever("dit_attn", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_DIT", "OPENFOLD3_OPT_DIT_MIN_TOKENS", "OPENFOLD3_OPT_DIT_CORE"),
           "the diffusion transformer's attention with pair bias (24 blocks x 200 steps, AdaLN AttentionPairBias) on the core's flash kernel (opt_core.kernels.dtk_kernels.flash_bias_attn: the [S,H,N,N] logits never materialised; fp32 softmax statistics; all samples in one call of the core's pair-bias attention — the core's ONE provider opt_core.kernels.apb asked by the family's word (cells/apb_word.py: the line's tier word OPENFOLD3_OPT_APB_TIER=fast|big, i.e. the measured row per capability / dtype / geometry / token bucket / eager-or-graph timing, or the caller's OPENFOLD3_OPT_APB_WORD), the direct entry opt_core.attn.apb_core when no word is set; OPENFOLD3_OPT_DIT_CORE=dtk asks the provider's per-sample dtk_loop row —, the shared pair bias read in place from the pair cache's head-major buffer) inside the sampler's CUDA graph; a size gate (opt_core.attn.size_gate, min tokens OPENFOLD3_OPT_DIT_MIN_TOKENS default 256) below which the stock core runs by design; other shapes = named fallback",
           "[openfold3-opt/dit_attn] installed", "opt/openfold3_opt/cells/dit_attn.py", ("fast", "big/resident"), T_MODEL),
    Lever("dit_glue", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_DIT_GLUE", "OPENFOLD3_OPT_DIT_GLUE_MIN_TOKENS", "OPENFOLD3_OPT_DIT_GLUE_CORE", "OPENFOLD3_OPT_DIT_GLUE_ROWS"),
           "the token diffusion transformer block (DiffusionTransformerBlock.forward: 24 blocks x 200 steps over the samples) as ONE schedule of fused pieces instead of "
           "~130 kernels: AdaLN on the core's ln_modulate with the conditioning rows periodic over the samples (never expanded), ONE q|k|v|gate GEMM, the pair-bias "
           "attention on its four column slices in place (the core's pair_bias_attention entry when carried, else dtk_kernels.flash_bias_attn; the pair cache's bias read "
           "in place), ONE output GEMM, the adaLN-zero gate + residual in one row kernel, the conditioned SwiGLU transition as one [a|b] GEMM + swiglu + GEMM + "
           "gate/mask/residual kernel; the conditioned transition and AdaLN OUTSIDE that schedule (the atom transformer's 3+3 blocks x 200 steps, fp32, and any "
           "refused token block) served class-wide by the same row schedules (OPENFOLD3_OPT_DIT_GLUE_ROWS=all|block, default all) "
           "(opt_core.of3_sampler.dit_glue + dit_rows, bound by cells/dit_glue.py); refuses by name -> the stock block / method, counted",
           "[openfold3-opt/dit_glue] installed", "opt/openfold3_opt/cells/dit_glue.py", ("fast",), T_MODEL),
    Lever("apb_trunk", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_APB_TRUNK", "OPENFOLD3_OPT_APB_TRUNK_MIN_TOKENS", "OPENFOLD3_OPT_APB_TRUNK_HIGH_PRECISION", "OPENFOLD3_OPT_APB_TRUNK_SCOPE", "OPENFOLD3_OPT_APB_TRUNK_PRODUCER"),
           "the pairformer single track's attention with pair bias (AttentionPairBias of the 48-block trunk per recycle pass and of the confidence head's "
           "pairformer per sample; not the diffusion transformer's AdaLN instances): the pair bias produced head-major (the module's LayerNorm(z), then one "
           "W_z @ LN(z)^T GEMM writing the [H,N,N] planes directly: no [N,N,H] intermediate, no permute pass, no per-call relayout) and the attention as ONE "
           "launch of the core's pair-bias kernel (opt_core.attn.apb_core: the strided projection views in place, the key mask by name, the sigmoid gate "
           "fused, the [H,N,N] logits never materialised); the bias producer is the core's fused LayerNorm+projection kernel (opt_core.kernels.ln_proj: "
           "one read of z, the planes written head-major into 16-byte-aligned rows; OPENFOLD3_OPT_APB_TRUNK_PRODUCER=mm keeps the module's LN + one GEMM, "
           "another lever may register its own through cells/apb_trunk.use_producer); refuses by name -> the stock forward, counted. Attribution knobs: "
           "OPENFOLD3_OPT_APB_TRUNK_HIGH_PRECISION=override|honour (default honour: a call asking use_high_precision_attention keeps the stock forward), "
           "OPENFOLD3_OPT_APB_TRUNK_SCOPE=trunk+confidence|trunk (default trunk+confidence; `trunk` runs the confidence head's calls on the stock forward, counted scope:confidence)",
           "[openfold3-opt/apb_trunk] installed", "opt/openfold3_opt/cells/apb_trunk.py", ("fast", "big/resident"), T_MODEL),
    Lever("templ_embed", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_TEMPL_EMBED",),
           "the template embedder around the template pair stack (TemplateEmbedderAllAtom, once per trunk pass over T templates x N x N): the eight bias-free "
           "feature Linears + linear_z(LayerNorm(z)) as ONE kernel writing the stack's [T,N,N,64] input (fp32 accumulation, one bf16 rounding) and the "
           "mean over templates / relu / linear_t as ONE kernel (a stride-0 template axis read in place) — opt_core.of3_trunk.templ_embed on the core's "
           "templ_embed kernels, bound by cells/templ_embed.py; the stack itself untouched; bf16 autocast, batch 1, any template count / token count / "
           "chain layout; refuses by name -> the stock forward, counted",
           "[openfold3-opt/templ_embed] installed", "opt/openfold3_opt/cells/templ_embed.py", ("fast", "big/resident"), T_MODEL),
    Lever("token_agg", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_TOKEN_AGG",),
           "the atom -> token mean of the atom-attention encoder (atomize_utils.aggregate_atom_feat_to_tokens: a full-shape int64 index + a float-atomic scatter-add per "
           "denoiser step) as ONE deterministic segment-reduce kernel over each token's contiguous atom run (dtk_kernels.seg_reduce; the runs derived once per rollout, "
           "refreshed in place: opt_core.of3_sampler.rollout_memo); batch-1 inference, any sample / token / atom count; other layouts refused by name -> stock",
           "[openfold3-opt/token_agg] installed", "opt/openfold3_opt/cells/token_agg.py", ("fast", "big/resident"), T_MODEL),
    Lever("atom_window", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_ATOM_WINDOW", "OPENFOLD3_OPT_ATOM_WINDOW_PRECISION", "OPENFOLD3_OPT_ATOM_WINDOW_INV"),
           "the diffusion sampler's sequence-local atom attention (AtomAttentionEncoder / Decoder: 3 + 3 DiffusionTransformerBlocks x 200 steps over "
           "the samples) on the core's two fused kernels (opt_core.kernels.atom_window: LayerNorm + both AdaLN modulations from per-atom conditioning "
           "rows + the q|k|v|gate projections in one pass with k / v once per atom instead of on gathered 128-key window copies; then, per query block "
           "and sample, the shifted key window read in place, pair bias + validity mask, fp32 softmax, p v, sigmoid gate, W_o, AdaLN-Zero gate and "
           "residual in one kernel — TF32 round-to-nearest dots, fp32 in/out); the step-invariant conditioning / bias / mask operands once per rollout "
           "(opt_core.of3_sampler.rollout_memo); refuses by name -> the stock block, counted",
           "[openfold3-opt/atom_window] installed", "opt/openfold3_opt/cells/atom_window.py", ("fast", "big/resident"), T_MODEL),
    Lever("atom_hoist", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_ATOM_HOIST", "OPENFOLD3_OPT_ATOM_HOIST_MAX_GB", "OPENFOLD3_OPT_ATOM_HOIST_INV"),
           "the atom path's step-invariant work once per rollout instead of per denoiser step: AtomAttentionEncoder.get_atom_reps (c_l, p_lm: reference embedding, pair "
           "activations incl. LN+linear of the trunk pair, pair MLP), the atom transformers' LN_z(p_lm) and per-block pair-bias linear_z, the block-index utilities — "
           "memoised per rollout in address-stable buffers refreshed in place (opt_core.of3_sampler.rollout_memo; under the CUDA-graphed step the first denoiser call of each "
           "rollout runs eager to refresh them); every stock expression kept: bitwise the line without it; publishes per block the `_of3opt_atom_inv` buffers",
           "[openfold3-opt/atom_hoist] installed", "opt/openfold3_opt/cells/atom_hoist.py", ("exact", "fast"), T_MODEL),
    Lever("castcache", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_CASTCACHE",),
           "openfold3's Linear / LayerNorm primitives cast their fp32 weight and bias to bf16 on every bf16 call (two copy kernels per call); the bf16 copies "
           "memoised per module (key: the fp32 tensors' storage + version — reloaded, moved or edited weights refresh it) and handed to the same F.linear / "
           "F.layer_norm call: bitwise the line without it; ~336 MiB of bf16 weight copies (opt_core.of3_trunk.castcache)",
           "[openfold3-opt/castcache] installed", "opt/openfold3_opt/cells/castcache.py", ("exact", "fast"), T_MODEL),
    Lever("exactln", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_EXACTLN", "OPENFOLD3_OPT_EXACTLN_WORD"),
          "the `LayerNorm` primitive (the one class behind every LayerNorm of the trunk, the MSA / template stacks, the diffusion transformer and the heads) bound to the shared core's "
          "LayerNorm provider (opt_core.kernels.ln: one provider over the NVRTC replica of ATen's kernel `exactln`, the Triton rows and the measured cell table) by the word `exact`: per call "
          "class (form, width, row count, token bucket, eager | graph, stack) the provider names an exact-class row only where it is measured bitwise F.layer_norm AND at least as fast as "
          "ATen on the running card and stack, else ATen by name (the statement runs, counted); every exact-class row's first call per signature proven torch.equal against the statement "
          "in-process (a differing signature retires to the statement, named); the replica's kernels compile and self-check in a background thread at install; other dtypes / devices / "
          "non-contiguous / autocast-enabled calls keep the statement, counted (cells/exactln.py, of3_exactln.py; knob OPENFOLD3_OPT_EXACTLN_WORD=<tier word | provider row>)",
          "[openfold3-opt/exactln] installed", "opt/openfold3_opt/cells/exactln.py; opt/openfold3_opt/of3_exactln.py; opt/openfold3_opt/hooks/cells/sitecustomize.py", ("exact",), T_MODEL),
    Lever("ln_provider", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_LN_PROVIDER", "OPENFOLD3_OPT_LN_TIER", "OPENFOLD3_OPT_LN_WORD"),
          "the same LayerNorm binding asked with the line's TIER word — `fast` on the fast line (the provider's fastest measured row inside each cell's identity band: the Triton rows "
          "`fastln` / `ln_rows` on the N²-row pair and S·N-row MSA norms, the replica or ATen on the N-row operands, capture-aware — a call met during CUDA-graph capture asks the graph "
          "cell), `big` on the memory line (the lowest-peak row among those); a tolerance row's first call per signature witnessed against the statement (rel-RMS bound, else the "
          "signature retires to the statement by name), exact-class rows proven bitwise; cells naming ATen run the statement, counted (cells/ln_provider.py, of3_exactln.py; knob "
          "OPENFOLD3_OPT_LN_WORD=<tier word | provider row>)",
          "[openfold3-opt/ln_provider] installed", "opt/openfold3_opt/cells/ln_provider.py; opt/openfold3_opt/of3_exactln.py; opt/openfold3_opt/hooks/cells/sitecustomize.py", ("fast", "big/resident"), T_MODEL),
    Lever("triatt_exact", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_TRIATT_EXACT",),
           "the cuEquivariance triangle-attention call of the exact line's pair stacks (primitives.attention.triangle_attention, every call above the library's 100-token threshold) bound to the core's triangle-attention provider on word exact, which answers with the stock library call itself on every card (where the cell table says so the same library kernel is issued one head at a time, exact_headsplit, identical bits, less transient memory); bitwise the line without it; refusals and fallbacks by name on the LEVER line",
           "[openfold3-opt/triattn] triatt_exact: installed", "opt/openfold3_opt/cells/triatt_exact.py; opt/openfold3_opt/of3_triattn.py", ("exact",), T_MODEL),
    Lever("trimul_exact", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_TRIMUL_EXACT",),
           "the pair stacks' triangle multiplicative updates (outgoing + incoming) served by the core's ONE triangle-multiplication provider (opt_core.kernels.trimul, cell table TRIMUL_CELLS.json, binding opt_core.trimul.by_word) on word exact: per call class the exact-class row the table measured bitwise and at or above the line's op — tmk3_exact (the cuEquivariance TriMul's operations in order on the core's kernels) at c_z 128 up to 400 tokens and at c_z 64 (template pair stack) at every size, H100 and A100; the line's own cuEquivariance call everywhere else, by name; a kernel row serves a class only after its first eager call proved torch.equal against the line's statement (a class whose bits differ is refused for the process, named)",
           "[openfold3-opt/trimul_exact] installed", "opt/openfold3_opt/cells/trimul_exact.py", ("exact",), T_MODEL),
    Lever("trimul_form", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_TRIMUL_EXACT_FORM",),
           "rides on trimul_exact: the TriMul call classes whose statement is the MODULE's own inference path (use_cueq_triangle_kernels False at the call: the template pair stack, c_z 64, on every input; every pair block when the runner configuration switches the library off) served by the core's trimul provider asked for the exact tier under this engine's form key (opt_core.kernels.trimul select(word=exact, form=of3_module) -> the row of3_form where a form cell proves it, refused by name elsewhere; >= 0.5.67.0: that statement issued whole-tensor -- the projections and the gate as the module's GEMMs once over the pair tensor, the contraction in upstream's 256-column blocks, LayerNorms on the core's exactln row) once the first call of each (class, token count) proved torch.equal against the module on that call's operands (a shape whose bits differ refuses its class for the process, named); left off, those classes run the module's statement; H100-tested",
           "[openfold3-opt/trimul_form] armed", "opt/openfold3_opt/cells/trimul_form.py; opt/openfold3_opt/cells/trimul_exact.py", (), T_MODEL),                # off by default on every line: switch it on with the env word
    Lever("transition_exact", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_TRANSITION_EXACT",),
           "the exact line's SwiGLU transitions (layers.transition.SwiGLUTransition: the pairformer, MSA-module and confidence pair-stack transitions, c 128 / hidden 512, bf16 under the trunk's autocast; the template pair stack's c 64) bound to the core's ONE transition provider (opt_core kernels.transition) by the tier word exact under this engine's statement form (form liger: OpenFold3 0.4.1's LigerSiLUMulFunction silu*b, one rounding), fed the module's own LayerNorm output: where the provider's table names a kernel row (v1:liger: everything after the LayerNorm one Triton kernel) the exact-tier winner for the cell on this process's stack - bitwise against the statement there, from the stack's token floor up - the call launches it; a stock winner, a cell without a vouch on this stack or row count, no cell (the c 384 single transition, c 64 x 256), fp32 calls and module variants run the module's statement, counted with the provider's word; no kernel, tile table or floor in the kit: bitwise the line without it (openfold3_opt.of3_transition)",
           "[openfold3-opt/transition_exact] installed", "opt/openfold3_opt/cells/transition_exact.py; opt/openfold3_opt/of3_transition.py", ("exact",), T_MODEL),
    Lever("apb_hoist", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_APB_HOIST",),
           "AttentionPairBias._prep_bias' key-mask bias `inf * (mask - 1)` memoised per stack call (the mask is constant over the 48 pairformer blocks of a "
           "recycle pass) instead of rebuilt per block; the pair-bias term computed as stock: bitwise (opt_core.of3_trunk.apb_hoist; on the fast line apb_trunk "
           "serves the pairformer / confidence instances and passes the rest here)",
           "[openfold3-opt/apb_hoist] installed", "opt/openfold3_opt/cells/apb_hoist.py", ("exact", "fast", "big/resident"), T_MODEL),
    Lever("trunk_graph", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_TRUNK_GRAPH", "OPENFOLD3_OPT_TRUNK_GRAPH_NMAX"),
           "PairFormerStack.forward (the 48-block trunk stack per recycle pass, the confidence head's 4-block stack per call; launch-bound below ~1000 tokens) "
           "captured into a CUDA graph per (instance, input signature, flags, numerics mode) at its second call and replayed after, first call eager, one "
           "process pool; token count above OPENFOLD3_OPT_TRUNK_GRAPH_NMAX (1024), grad, CPU, nested capture, DS4Sci attention, lma or a failed capture run the "
           "stack's own forward by name: bitwise (opt_core.of3_trunk.trunk_graph)",
           "[openfold3-opt/trunk_graph] installed", "opt/openfold3_opt/cells/trunk_graph.py", ("fast",), T_MODEL),
    Lever("post_release", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_POST_RELEASE", "OPENFOLD3_OPT_POST_RELEASE_NTOK", "OPENFOLD3_OPT_POST_RELEASE_MIN_FREE_GB"),
           "the graphed lines' item-boundary memory release at large token counts (>= OPENFOLD3_OPT_POST_RELEASE_NTOK, 2400, or free memory < "
           "OPENFOLD3_OPT_POST_RELEASE_MIN_FREE_GB, 16): the graphed step's generation + pool + gather tables, the pair cache's static buffers, trunk_graph's "
           "graphs, cuBLAS workspaces and cached-free allocator segments released right after the sampler returns (before the confidence head's per-sample "
           "pair batch: at 3000 tokens x 5 samples the item otherwise fails there) and after the item's forward; MiB freed per source on the census; the next "
           "item re-captures; outputs untouched: bitwise (opt_core.of3_sampler.post_release)",
           "[openfold3-opt/post_release] installed", "opt/openfold3_opt/cells/post_release.py", ("exact", "fast", "big/resident"), T_MODEL),
    Lever("tuner_guard", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_TUNER_GUARD",),
           "the engine's per-stack chunk-size tuner compares the argument record of its last tuning with the next chunked call's and raises when "
           "they differ in structure (the confidence stack's batched call at or below its 750-token per-sample cutoff vs its per-sample calls above: "
           "tensor ranks differ) — one process predicting a query of <= 750 tokens and then a larger one failed every larger query on the kernels-off "
           "runner configuration; the kit answers such a comparison 'changed' so the tuner re-tunes, counted (opt_core.of3_trunk.tuner_guard)",
           "[openfold3-opt/tuner_guard] installed", "opt/openfold3_opt/cells/tuner_guard.py", ("fast", "big/resident"), T_MODEL),
    Lever("sync_hoist", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_SYNC_HOIST",),
           "the recycle loop's three per-pass GPU read-backs served once per trunk call: the MSA depth draw kept without its host read (the "
           "subsampling bounds are equal: the draw can only be their value), the MSA valid / invalid row indices computed from the query's MSA mask "
           "at the first pass and reused, the trunk-kernels add-on's by-value distinct-template verdict decided at the first template-stack call and "
           "served after it (an instance-level forward over the add-on's class-level wrap) — the host no longer stalls at every recycle boundary "
           "behind the previous pass's kernels; same kernels, operands and generator draws: bitwise; configurations outside the served statements "
           "run the engine's own by name (opt/openfold3_opt/cells/sync_hoist.py; a core-lift candidate)",
           "[openfold3-opt/sync_hoist] installed", "opt/openfold3_opt/cells/sync_hoist.py", ("exact", "fast", "big/resident"), T_MODEL),
    Lever("fastjson", "cells", OUTPUT, EXACT, ("OPENFOLD3_OPT_FASTJSON", "OPENFOLD3_OPT_FASTJSON_ROUTE"),
           "the engine's output writer renders each sample's aggregated and full confidence JSON (plddt per atom, pde / pae per token pair; "
           "json.dumps(indent=4, cls=NumpyEncoder): CPython's pure-Python indent encoder) through a subclass of its encoder class whose encode() builds "
           "the same characters row by row at C level (str.join over float.__repr__, NaN / Infinity as json spells them, the engine's own array "
           "conversion) — every output file identical to the unpatched writer's; the model forward untouched, the item's wall time after it shorter; "
           "a structure the port does not reproduce is the stock encoder's call, counted; kept only when its probe text equals the stock class's at install "
           "(cells/of3_fastjson.py; OPENFOLD3_OPT_FASTJSON_ROUTE=stock: the class installed, every call the stock encoder's)",
           "[openfold3-opt/fastjson] installed", "opt/openfold3_opt/cells/fastjson.py", ("exact", "fast", "big/resident", "big/tp"), T_WRITER),
    Lever("writer_overlap", "cells", OUTPUT, EXACT, ("OPENFOLD3_OPT_WRITER_OVERLAP", "OPENFOLD3_OPT_WRITER_OVERLAP_ROUTE"),
           "the engine's output writer callback (OF3OutputWriter.on_predict_batch_end -> write_all_outputs: per diffusion sample one mmCIF through "
           "biotite and two indented JSON texts, on the main thread between two forwards) re-stated so the callback's bookkeeping and the "
           "device-to-host copies the writer makes first stay where the engine runs them and the rendering + writing — the writer instance's own "
           "write_all_outputs on those host copies, every other writer patch (fastjson) applying unchanged — runs in one writer worker (a forked "
           "child process by default: no interpreter lock shared with the next forward's launches; a thread or inline by knob) while item k+1 is "
           "featurised and forwarded, at most two items in flight, every item drained and tallied into the callback's own counters before the "
           "engine's summary and at exit; write_features / write_latent_outputs / a distributed predict written inline by name, a worker that dies "
           "has its items re-written inline (never lost, counted) — every output file identical, the wall time between two forwards shorter, the "
           "model forward untouched (cells/of3_writer_overlap.py; OPENFOLD3_OPT_WRITER_OVERLAP_ROUTE=process|thread|sync)",
           "[openfold3-opt/writer_overlap] installed", "opt/openfold3_opt/cells/writer_overlap.py", ("exact", "fast", "big/resident"), T_WRITER),
    Lever("hostfeat", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_HOSTFEAT", "OPENFOLD3_OPT_HOSTFEAT_PARTS"),
           "host-side featurisation statements of the engine's data pipeline re-stated in numpy: the a3m deletion matrix + aligned letter matrix "
           "(core.data.io.sequence.msa.parse_a3m: a per-character Python loop over every MSA row, then translate + a row-by-row '<U1' fill) from the "
           "byte view of the rows and the MSA letter -> alphabet index mapping (core.data.resources.residues.map_str_array_to_idx_array: one full-array string comparison per alphabet letter over the '<U1' MSA matrix) through a 256-entry table filled by the engine's own function — "
           "integer / string arrays only, every feature identical (dtype, shape, elements: the kit tests against the engine's text); item 1's featurisation, "
           "which every process's first forward waits for, seconds shorter at the ladder sizes; a row outside the byte identity (non-ASCII, ragged) or an "
           "engine text the port does not re-state goes to the engine's statement by name (cells/of3_hostfeat.py; OPENFOLD3_OPT_HOSTFEAT_PARTS=a3m,msaidx)",
           "[openfold3-opt/hostfeat] installed", "opt/openfold3_opt/cells/hostfeat.py", ("exact", "fast", "big/resident", "big/tp"), T_MSA_IO),
    Lever("ckpt_mmap", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_CKPT_MMAP",),
           "the engine's checkpoint read (core.utils.checkpoint_loading_utils.load_checkpoint: torch.load of 2.3 GB of CPU-tagged fp32 tensors, "
           "unpickled into fresh host memory in every process before the model can load them) made through the same torch.load call with "
           "mmap=True — the storages memory-mapped instead of copied, the same dict of the same tensors handed to get_state_dict_from_checkpoint / "
           "load_state_dict; a directory checkpoint or a file torch cannot map is the engine's own call, counted (cells/of3_ckptmmap.py)",
           "[openfold3-opt/ckpt_mmap] installed", "opt/openfold3_opt/cells/ckpt_mmap.py", ("exact", "fast", "big/resident", "big/tp"), T_CKPT),
    Lever("postfwd_mem", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_POSTFWD_MEM", "OPENFOLD3_OPT_POSTFWD_MEM_MIB"),
           "the runner's confidence scoring after the forward (the engine's get_confidence_scores on the forward's outputs) without the "
           "[S, N, N, 64] probability tensors (pde, pae, every compute_ptm call) and the [S, N_atom, N_atom, 3] all-atom difference tensor of "
           "get_token_frame_atoms that set the process peak of every line after OpenFold3.forward returned (1200 tokens: +9 to +11 GiB above the "
           "forward's own peak): those statement groups run per (sample, row block) under a MiB budget (_MIB, default 256; 0 = the engine's "
           "statements) and the engine's full-size result tensors are assembled by copy, every long reduction on the engine's own tensors — "
           "bitwise; on big/resident it serves below the confidence gate (modes.CONF_MIN_TOKENS), the port's chunked scorer above it "
           "(openfold3_opt.of3_postfwd)",
           "[openfold3-opt/postfwd_mem] installed", "opt/openfold3_opt/cells/postfwd_mem.py; opt/openfold3_opt/of3_postfwd.py", ("exact", "fast", "big/resident"), T_MODEL),
    Lever("loader_workers", "cells", FORWARD, EXACT, ("OPENFOLD3_OPT_LOADER_WORKERS", "OPENFOLD3_OPT_LOADER_WORKERS_CAP"),
           "the engine's predict DataLoader forks data_module_args.num_workers featurisation workers (10) however few items the query set holds; "
           "the kit caps the predict loader at max(1, min(items, configured)) — the same worker featurises the same item with the same seed "
           "(in-order loader, per-worker seeding by index): byte-identical outputs, fewer forked interpreters (~5 GB host RSS each); "
           "_CAP=none = the engine's count (openfold3_opt.of3_loader)",
           "[openfold3-opt/loader_workers] installed", "opt/openfold3_opt/cells/loader_workers.py; opt/openfold3_opt/of3_loader.py", ("exact", "fast", "big/resident"), T_MODEL),
    Lever("tp_shard_s", "tp_rowpair", FORWARD, TOLERANCE, (),
           "tensor parallelism: the pair representation row-sharded across the ranks (one process per GPU, an NCCL group) — the trunk's pair stack, "
           "MSA module, template embedder, diffusion conditioning and confidence heads run on each rank's rows with ring-streamed / transposed peers "
           "(opt_core.mem.rowpair bound by opt/openfold3_opt/tp_rowpair); the LayerNorm 2^31-element guard; not bitwise vs the one-card routes on this tree",
           "[tp_rowpair hook r", "opt/openfold3_opt/tp_rowpair/__init__.py; opt/openfold3_opt/tp_rowpair/hook/sitecustomize.py; opt/openfold3_opt/tp_rowpair/env.py; opt/openfold3_opt/tp.py", ("big/tp",), T_MODEL),
    Lever("tp_triatt", "tp_rowpair", FORWARD, TOLERANCE, (),                       # on with the tp line: the flash_triattn kernel word (tp_rowpair/pairstack.TRIATT_KERNEL); the core's opt-out ROWPAIR_TRIATT_CORE=torch
           "the row-sharded pair stack's triangle attention on the core's row-block kernel dispatch (opt_core.mem.rowpair.triatt.attention_core kernel=<word>, ONE "
           "vocabulary): flash_triattn = the core's carried flash triangle-attention kernel per row batch; OpenFold3's "
           "torch attention statement is the dispatch's named per-call fallback (the core's LEVER name=F1.flash_triattn line: served / fallback / fallback_by; one "
           "[tp_rowpair] TRIATT line per rank, the manifest's tp block); a word the interpreter cannot serve is refused by name at bind. Not bitwise vs the torch statement",
           "[tp_rowpair] TRIATT kernel=", "opt/openfold3_opt/tp_rowpair/pairstack.py (triatt_kernel, triatt_fns over opt_core.mem.rowpair.triatt.attention_core); opt/openfold3_opt/tp.py (census)", ("big/tp",), T_MODEL),
    Lever("tp_trimul", "tp_rowpair", FORWARD, TOLERANCE, (),                       # on with the tp line: the fpf_v4 provider word (tp_rowpair/pairstack.TRIMUL_KERNELS); the core's opt-out ROWPAIR_TRIMUL_KERNELS=torch
           "the row-sharded pair stack's triangle multiplication on the core's fused row-block provider (opt_core.mem.rowpair.trimul_fused around OpenFold3's torch "
           "TriMulFns: the carried fpf_trimul_v4 kernels — K1 LayerNorm + one gated projection per launch written into the GEMM block, K3 LayerNorm-out + W_o + gate + "
           "residual per contraction tile; launch cells from the core's fpf_trimul_v4 table per (cc, triton); c_z = c_hidden in {128,256}; pairs below the provider's "
           "size gate (2048 tokens) and any unit the provider declines run the torch statements, counted by reason (the core's LEVER name=F2.trimul_rows line; one "
           "[tp_rowpair] TRIMUL line per rank, the manifest's tp block). Not bitwise vs the torch statements",
           "[tp_rowpair] TRIMUL kernels=", "opt/openfold3_opt/tp_rowpair/pairstack.py (trimul_kernels, trimul_fns over opt_core.mem.rowpair.trimul_fused); opt/openfold3_opt/cells/pairfused.py (trimul_weights); opt/openfold3_opt/tp.py (census)", ("big/tp",), T_MODEL),
    Lever("sample_loop", "tp_rowpair", FORWARD, TOLERANCE, (),                     # on with the tp line, no switch of its own
          "the diffusion roll-out and the confidence head one sample chunk at a time (opt_core.mem.sample_loop over opt_core.mem.ckpt's stock-order draws: "
          "the generator rewound per pass, every batch-shaped draw made at the S shape and sliced, per-pass outputs assembled on the host in stock order) — "
          "device memory independent of --num-diffusion-samples; one sample per pass on the row-sharded line",
          "[sample_loop] samples=", "opt/openfold3_opt/sample_loop.py; opt/openfold3_opt/tp_rowpair/model.py (rollout_rows)", ("big/tp",), T_MODEL),
    Lever("trimul_hostsnap", "offload", FORWARD, EXACT, ("OF3O_TRIMUL", "OF3O_ROWS"),
           "O1: triangle multiplication with the residual snapshot on the host — the pair update streamed in row blocks (OF3O_ROWS) from a pinned host copy instead of a second GPU copy of z",
           "[of3o]", "offload/of3o/of3_offload.py:176-270; offload/README.md", ("big/resident",), T_MODEL),
    Lever("triatt_lean", "offload", FORWARD, EXACT, ("OF3O_TRIATT",),
           "O1: the ending triangle attention on a lean transposed copy (one transient N×N buffer, released before the transition)",
           "[of3o]", "offload/of3o/of3_offload.py:272-324", ("big/resident",), T_MODEL),
    Lever("trans_inplace", "offload", FORWARD, EXACT, ("OF3O_TRANS",),
           "O1: the pair transition applied in place in row blocks (no full-size intermediate)",
           "[of3o]", "offload/of3o/of3_offload.py:325-354", ("big/resident",), T_MODEL),
    Lever("cond_once", "offload", FORWARD, EXACT, ("OF3O_COND", "OF3O_COND_ROWS"),
           "O1: the diffusion conditioning's pair path computed once per item in row blocks (OF3O_COND_ROWS) and cached for every sample and step",
           "[of3o]", "offload/of3o/of3_offload.py:413-481", ("big/resident",), T_MODEL),
    Lever("input_rows", "offload", FORWARD, EXACT, ("OF3O_INPUT", "OF3O_INPUT_ROWS", "OF3O_EMBED_ROWS"),
           "O1: the input embedder's relative-position pair features built in row blocks (OF3O_INPUT_ROWS) and the pairformer embedding in row blocks (OF3O_EMBED_ROWS) — no N×N×features intermediate",
           "[of3o]", "offload/of3o/of3_offload.py:356-411,824-882", ("big/resident",), T_MODEL),
    Lever("recycle_rows", "offload", FORWARD, EXACT, ("OF3O_RUN_TRUNK", "OF3O_RECYCLE_ROWS"),
           "O1: the recycling embedder's pair add in row blocks from the host snapshot (OF3O_RECYCLE_ROWS) inside the port's run_trunk",
           "[of3o]", "offload/of3o/of3_offload.py:534-652", ("big/resident",), T_MODEL),
    Lever("templ_host", "offload", FORWARD, EXACT, ("OF3O_TEMPL", "OF3O_TEMPL_HOST", "OF3O_TEMPL_ROWS", "OF3O_TEMPL_EMBED_ROWS", "OF3O_TEMPL_DEDUP"),
           "O1: template pair features kept on the host and embedded in row blocks (OF3O_TEMPL_ROWS / OF3O_TEMPL_EMBED_ROWS), duplicate templates de-duplicated",
           "[of3o]", "offload/of3o/of3_offload.py:946-1102", ("big/resident",), T_MODEL),
    Lever("conf_chunked", "offload", FORWARD, EXACT, ("OF3O_CONF_MODE", "OF3O_CONF_ROWS", "OF3O_CONF_MIN", "OF3O_CONF_TM_BACKEND"),
           "the confidence scorer over row blocks: PAE/PDE/pLDDT/TM from the pair logits in rows (OF3O_CONF_ROWS) with the BLOCKREDUCE TM backend (explicit; the native backend is a counted event, never a silent switch) — the consumer the row-block head (`confhead`) and the host logits (`logits_host`) require: the big lines carry it with them from modes.CONF_MIN_TOKENS tokens up (below: `stock` = upstream's scorer); on its own it moves no peak",
           "[of3o]", "offload/of3o/of3_offload.py:915-976; offload/of3o/of3o_confidence.py; offload/of3o/of3o_blockreduce.py", ("big/resident",), T_MODEL),
    Lever("confhead", "confhead", FORWARD, TOLERANCE, ("OPENFOLD3_OPT_CONFHEAD",),
           "the PAE/PDE confidence heads evaluated per scorer row block (opt/openfold3_opt/confhead.py: the heads' forward returns a RowBlockLogits — the pair representation stays, the [S,N,N,64] logits never exist; the head's LayerNorm + C->64 GEMM run on the block the chunked scorer slices) — the resident line's confidence-phase peak from modes.CONF_MIN_TOKENS tokens up (modes.conf_gate; requires conf_chunked)",
           "[openfold3-opt/confhead] installed", "opt/openfold3_opt/confhead.py; opt/openfold3_opt/hooks/confhead/sitecustomize.py", ("big/resident",), T_HEADS),
    Lever("bigln_guard", "offload", FORWARD, EXACT, ("OF3O_LNSAFE",),
           "LayerNorm over ≥ 2^31 elements in row blocks (the CUDA kernel's 32-bit index guard), scoped to the model's own LayerNorm instances",
           "[of3o]", "offload/of3o/of3_offload.py:1450-1508", ("big/resident",), T_MODEL),
    Lever("host_pool", "offload", FORWARD, EXACT, ("OF3O_PIN_BUDGET_GB", "OF3O_PIN_POLICY"),
           "the pinned host buffer pool the row levers stream through (one buffer per tag; OF3O_PIN_BUDGET_GB the pinned total; a buffer over it is pageable with a NOTE line and a count under OF3O_PIN_POLICY=census, the lines' word, refused by name under `strict`; a pinned allocation the host refuses raises by name)",
           "[of3o]", "offload/of3o/of3_offload.py:81-132", ("big/resident",), T_MODEL),
    Lever("chunk_pin", "offload", FORWARD, EXACT, ("OF3O_CHUNK",),
           "a fixed chunk plan from the environment for the stacks whose tuners the line disables (the big rows pin the plan through the runner yaml instead: offload/config/stock_predict_c16.yml)",
           "[of3o]", "offload/of3o/of3_offload.py:68-79,134-174", (), T_MODEL),
    Lever("alloc_expandable", "offload", FORWARD, EXACT, ("PYTORCH_CUDA_ALLOC_CONF",),
           "the expandable-segments CUDA allocator for the kit process (fragmentation headroom at large N; the effective setting is recorded in the census)",
           "", "offload/README.md; offload/of3o/of3_offload.py:1555-1612", ("fast", "big/resident"), T_MODEL),
]}

# The switch a carried hook reads that is a testing/measurement aid, never a lever (det.py applies it under --det 1).
AIDS: Dict[str, str] = {
    "OF3_DETERMINISTIC": "the deterministic reference configuration: torch deterministic algorithms + CUBLAS_WORKSPACE_CONFIG=:4096:8 (fast_inference/of3_levers/sitecustomize.py)",
}


# The canonical strategy id (opt_core/STRATEGIES.json `canonical[].id`; opt_core.report.lever_line validates it, an alias is refused) of the
# levers whose LEVER line names one: the confidence-phase levers of the big lines. report.lever_lines reads it; the tensor-parallel lever's id
# comes from opt_core.mem.ngpu there.
STRATEGY: Dict[str, str] = {
    "confhead": "F7.chunked_eval",          # the head evaluated per scorer row block: chunked evaluation of the confidence phase
    "conf_chunked": "F7.chunked_eval",      # the scorer over row blocks,       # the O2 logits resident in pinned host rows
}


def by_mode(mode_line: str) -> Tuple[str, ...]:
    """Lever names that name `mode_line` (e.g. "fast", "exact", "big/resident") in their `modes` field."""
    return tuple(n for n, l in LEVERS.items() if mode_line in l.modes)


# ---- GPU-class support (opt_core.arch: the one registry every lever consults; README §Applicability's GPU-classes column is card_table()) ----
ARCH_PREFIX = "LOCAL.openfold3."                 # kit-local lever ids in the arch registry (shared strategies are declared by their family module)
TESTED_SM = ("sm90",)                         # the class every test record of this kit was taken on (H100)
MIN_SM = "sm80"                                  # the carried add-ons state sm80+ (stack.py MIN_CC; bf16/TF32 tensor-core paths)
# levers that have also been exercised on sm80 (A100 80 GB, configs/a100.env; the pinned stack with the DS4Sci op built for sm_80): the levers of
# the exact (cueq), fast and big·tp (--n_gpu 2) lines and the ds4sci route — exact --det 1 bitwise == off --det 1 on small test inputs with
# every exact-line cell applied (castcache, apb_hoist, atom_hoist, post_release, postfwd_mem, loader_workers, fastjson); the fast line's pair
# cells served from the 8.0 rows of the core tables, its sampler / trunk cells (dit_glue, token_agg, atom_window with its 8.0 tile row,
# atom_hoist, apb_trunk, templ_embed, trunk_graph up to its nmax, tuner_guard) serving at 400 / 800 / 1200 tokens, the allocator policy applied;
# big --n_gpu 2 complete on two cards. sync_hoist (added after those records), the big resident / streamed / of3tp levers and the opt-in
# extras keep sm90 as their only tested class (sm80 stays their untested floor).
A100_TESTED = ("exactln", "ln_provider", "trimul_exact", "fast_init", "cuda_graphs", "graphs_strict", "templ_distinct", "paircache", "trimul_cueq", "trimul_v4",
               "triatt_block", "pair_transition", "rollout_bf16", "dit_attn", "dit_glue", "token_agg", "atom_window", "atom_hoist", "apb_trunk",
               "templ_embed", "castcache", "apb_hoist", "trunk_graph", "post_release", "tuner_guard", "alloc_expandable", "postfwd_mem",
               "loader_workers", "fastjson", "triatt_exact", "transition_exact", "triatt_provider", "tp_shard_s", "sample_loop", "ds4sci_line",
               "trimul_hostsnap", "triatt_lean", "trans_inplace", "cond_once", "input_rows", "recycle_rows", "templ_host", "bigln_guard", "host_pool",
               "sync_hoist")                                               # the resident line's port units and sync_hoist: run as the composed one-GPU `big` line
                                                                           # tokens; conf_chunked / confhead (from 2048 tokens) keep sm80 as their untested floor


def tested_sm(name: str) -> tuple:
    """The GPU classes lever `name` has test records on, ascending: sm80 (A100) for the A100_TESTED levers, then TESTED_SM (sm90) for all."""
    return (("sm80",) if name in A100_TESTED else ()) + TESTED_SM
# levers whose implementation is an image-bound compiled op: the class set is the image's (named, not excluded — another image of the same stack runs them)
ARCH_NOTES = {
    "trimul_cueq": "cuequivariance_wheel_ships_per_arch_kernels",
    "trimul_v4": "triton_kernel_jit_compiled_per_sm;cells_keyed_cc|triton(opt_core kernels/fpf_trimul_v4/table.json)",
    "triatt_block": "triton_kernels_jit_compiled_per_sm;core_cell_table(opt_core.attn.pair_fused_cells.json)",
    "pair_transition": "triton_kernel_jit_compiled_per_sm;core_cell_table(opt_core.attn.pair_fused_cells.json)",
    "rollout_bf16": "torch_autocast_bf16;no_kernel_of_its_own",
    "dit_attn": "triton_kernel_jit_compiled_per_sm(apb_attn|dtk_kernels.flash_bias_attn)",
    "apb_trunk": "triton_kernel_jit_compiled_per_sm(apb_attn)",
    "dit_glue": "triton_kernels_jit_compiled_per_sm(dtk_kernels.ln_modulate|swiglu|gate_residual|flash_bias_attn)",
    "token_agg": "triton_kernel_jit_compiled_per_sm(dtk_kernels.seg_reduce)",
    "atom_window": "triton_kernels_jit_compiled_per_sm(atom_window.ln_qkvg|window_attn)",
    "templ_embed": "triton_kernels_jit_compiled_per_sm(templ_embed)",
    "atom_hoist": "engine_statements_memoised;no_kernel_of_its_own",
    "castcache": "engine_statements_memoised;no_kernel_of_its_own",
    "exactln": "nvrtc_kernel_jit_compiled_per_sm(opt_core.kernels.ln exactln: cubin cached under MODEL_OPT_JIT_ROOT/exactln when writable, else compiled per process at install; variant triton jit_compiled_per_sm);core_cell_table(opt_core.kernels.ln LN_CELLS.json cc 9.0|8.0)",
    "ln_provider": "nvrtc_kernel_jit_compiled_per_sm(opt_core.kernels.ln exactln, as exactln) + triton_kernels_jit_compiled_per_sm(opt_core.kernels.ln rows fastln / ln_rows: Triton's own cache under TRITON_CACHE_DIR, compiled per process otherwise; each signature's first call off any capture)",
    "triatt_provider": "core_cell_table(opt_core.kernels.triattn.TRIATTN_CELLS.json:cc9.0+8.0);triton_rows_jit_compiled_per_sm",
    "trimul_provider": "core_cell_table(opt_core.kernels.trimul.TRIMUL_CELLS.json:cc9.0+8.0);triton_rows_jit_compiled_per_sm(tmk3,esm_v5_fwd:tl.make_tensor_descriptor>=triton3.6);kit_row_v4_per_cc_cells",
    "triatt_exact": "core_cell_table(opt_core.kernels.triattn.TRIATTN_CELLS.json);library_call_by_name(per_head_where_the_cell_says)",
    "trimul_exact": "core_cell_table(opt_core.kernels.trimul.TRIMUL_CELLS.json:cc9.0+8.0);tmk3_exact_triton+cublas(bit_proven_per_class);line_by_name_elsewhere",
    "trimul_form": "core_exact_tier_form_of3_module->row_of3_form(opt_core.kernels.trimul.of3_form:cublas_gemms+aten_or_exactln_layernorm,no_arch_specific_kernel;form_cells_cc9.0);bit_proven_per_class_and_shape_in_process;module_by_name_on_a_differing_shape",
    "transition_exact": "core_transition_provider(opt_core.kernels.transition:tier_word_exact,form_liger;cells+floors=TRANSITION_CELLS.json:cc9.0+8.0,per_stack_vouch);triton_kernel_jit_compiled_per_sm;stock_winner_or_unvouched_cell_runs_the_module_by_name;bitwise_checked_sm90_sm80",
    "apb_hoist": "engine_statements_memoised;no_kernel_of_its_own",
    "trunk_graph": "cuda_graph_capture_of_the_engine_stack;no_kernel_of_its_own",
    "post_release": "allocator_and_graph_state_release;no_kernel_of_its_own",
    "tuner_guard": "engine_tuner_comparison_guarded;no_kernel_of_its_own",
    "sync_hoist": "recycle_loop_readbacks_served_once_per_trunk_call;no_kernel_of_its_own",
    "fastjson": "host_side_json_text_rendering;no_kernel_of_its_own;no_gpu_code",
    "writer_overlap": "host_side_output_writing_overlapped_with_the_next_item;no_kernel_of_its_own;no_gpu_code",
    "hostfeat": "host_side_featurisation_in_numpy;no_kernel_of_its_own;no_gpu_code",
    "ckpt_mmap": "host_side_checkpoint_read;no_kernel_of_its_own;no_gpu_code",
    "postfwd_mem": "engine_scoring_statements_blockwise;no_kernel_of_its_own",
    "loader_workers": "engine_dataloader_worker_count;no_kernel_of_its_own",
    "tp_shard_s": "collectives_plus_the_engine_statements;chunk_plan_is_the_80GB_card_table(tp.CHUNK_PLAN)",
    "sample_loop": "engine_statements_on_fewer_samples;no_kernel_of_its_own",
}


# the exact/ds4sci line's stack requirement (a runner-yaml route, not a registry lever): DeepSpeed's evoformer attention op is compiled per image
ARCH_LINES = {"ds4sci_line": "deepspeed_evoformer_attn_op_compiled_per_image:img_freeze_ds4sci=sm90_only;img_freeze_ds4sci_sm80_100=sm80+sm90+sm100+PTX"}


def declare_arch() -> dict:
    """Declare every lever of this kit (and the ds4sci line's op) in opt_core.arch (idempotent: the same content twice is a no-op): tested on
    tested_sm(name) (sm90 for all, + sm80 for A100_TESTED), floor MIN_SM, no exclusions known; the note names an image/JIT dependence where
    there is one. Returns {id: Support}."""
    from opt_core import arch
    out = {}
    for name in LEVERS:
        out[name] = arch.declare(ARCH_PREFIX + name, certified=tested_sm(name), min_sm=MIN_SM, note=ARCH_NOTES.get(name))
    for name, note in ARCH_LINES.items():
        out[name] = arch.declare(ARCH_PREFIX + name, certified=tested_sm(name), min_sm=MIN_SM, note=note)
    return out


def card_table(sms=None) -> dict:
    """``{lever: {sm: word}}`` for this kit's levers over the arch registry's classes (README §Applicability's GPU-classes column)."""
    from opt_core import arch
    declare_arch()
    return {name[len(ARCH_PREFIX):]: row for name, row in arch.card_table([ARCH_PREFIX + n for n in list(LEVERS) + list(ARCH_LINES)], sms).items()}
