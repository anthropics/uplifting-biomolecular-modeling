"""Lever registry: metadata for every optimization the modes compose, keyed by the switches the add-ons' own hooks read.

No values live here: the switches and their values are the mode lines in modes.py (quoted from the kit READMEs); each entry names the
kit file and line the lever was read from, the switch it answers to, its numerics class, the stderr line the kit prints when it is
installed and the stock module whose first import makes the kit's finder install it (`target`: the moment the lever's own record —
`stack.lever_applied()` reads the add-ons' flags, state dicts and installed lists — can say applied or unavailable; before that import the
lever is pending). `tests/test_modes.py` covers the mode table `registry.LEVERS` is keyed against; `tests/test_kit_paths.py` locks
every `source` here against a file that exists in the carried kit.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

FORWARD = "forward"
OUTPUT = "output"                                                   # the phase after the model forward: the engine's output writer (a lever here moves the item's wall time, not the model forward)
EXACT, TOLERANCE = "exact", "tolerance"
T_LINEAR = "openfold3.core.model.primitives.linear"                 # fast_inference/of3_levers/sitecustomize.py:3,41-44
T_DIFFUSION = "openfold3.core.model.structure.diffusion_module"     # fast_inference/of3_levers/sitecustomize.py:4,45-47
T_MODEL = "openfold3.projects.of3_all_atom.model"                   # trunk_kernels/of3t_hook/sitecustomize.py:60; hooks/cells/sitecustomize.py
T_WRITER = "openfold3.core.runners.writer"                       # opt/openfold3_ob0_opt/fastjson.py (installed at activation; patched when the engine imports its writer)
T_CKPT = "openfold3.core.utils.checkpoint_loading_utils"        # opt/openfold3_ob0_opt/ckpt_mmap.py (installed at activation; patched when the runner imports its checkpoint reader)
T_MSA_IO = "openfold3.core.data.io.sequence.msa"               # opt/openfold3_ob0_opt/hostfeat.py (installed at activation; patched when the engine imports its MSA reader, before the DataLoader forks)
T_HEADS = "openfold3.core.model.heads.prediction_heads"          # opt/openfold3_ob0_opt/hooks/confhead/sitecustomize.py (_TARGET)
T_HEAD_MODULES = "openfold3.core.model.heads.head_modules"       # opt/openfold3_ob0_opt/conf_dtype.py (TARGET: AuxiliaryHeadsAllAtom.forward)


@dataclass(frozen=True)
class Lever:
    name: str
    kit: str                                   # the add-on's key in the modes table; the package-hooks key for a lever the package ships behind its own hook (confhead), or stack.PACKAGE for one the activation installs (conf_dtype)
    cls: str                                   # forward
    tier: str                                  # exact | tolerance
    env_keys: Tuple[str, ...]                  # the switch(es) the kit reads for it (names only; values in modes.LINES)
    description: str
    marker: str                                # the stderr line (prefix) the kit prints when the lever is installed; "" when it prints none
    source: str                                # where the switch was read (kit file:line)
    target: str = ""                           # the stock module whose import triggers the kit's install of this lever ("" = not applied by a hook)

    @property
    def modes(self) -> Tuple[str, ...]:
        """The house lines that carry this lever (modes.LINES), spelled `mode` or `mode/line`; derived from the one mode table, never stored here."""
        from .modes import LINES
        keys = [key for key, ln in LINES.items() if self.name in ln.levers]
        return tuple(m if ln_name is None else f"{m}/{ln_name}" for (m, ln_name) in keys)


LEVERS: Dict[str, Lever] = {l.name: l for l in [
    Lever("fast_init", "fast_inference", FORWARD, EXACT, ("OF3_FAST_INIT",),
           "U0: skips the CPU-side weight initialisation the checkpoint load overwrites — the init functions and Linear.reset_parameters that upstream's own skip_random_init leaves (model construction ~3.2 s -> ~0.2 s per process)",
           "[of3_levers] fast-init enabled", "fast_inference/README.md:10; of3_levers/sitecustomize.py:41-44", T_LINEAR),
    Lever("cuda_graphs", "fast_inference", FORWARD, EXACT, ("OF3_CUDA_GRAPHS",),
           "C3: captures one denoising step of the diffusion sampler per input shape and replays it for the other steps and for later items whose step reads the same shapes (OPENFOLD3_OB0_OPT_GRAPHS_KEEP; census kept=); memoised, per-item checked gather tables",
           "[of3_graphs]", "fast_inference/README.md:12; of3_levers/sitecustomize.py:45-47; of3_levers/of3_graphs.py", T_DIFFUSION),
    Lever("graphs_strict", "fast_inference", FORWARD, EXACT, ("OF3_GRAPHS_STRICT",),
           "capture failure raises instead of the eager fallback (the graphed arm is proven graphed: `invariants:` reads fallbacks=0)",
           "", "of3_levers/of3_graphs.py:328", T_DIFFUSION),
    Lever("templ_distinct", "trunk_kernels", FORWARD, EXACT, ("OF3T_TEMPL_DISTINCT",),
           "an untemplated chain's 4 identical dummy templates run through the template pair stack once, re-expanded before the stock reduction; templated chains and the template-module offload loop take the stock path, named per item",
           "[of3t_levers] installed:", "trunk_kernels/README.md:13; of3t_hook/sitecustomize.py:31,77; of3t_hook/of3t_levers.py", T_MODEL),
    Lever("paircache", "trunk_kernels", FORWARD, EXACT, ("OF3T_PAIRCACHE",),
           "D1: the step-invariant conditioned pair representation and the 24 per-block pair biases of the token diffusion transformer computed once per rollout, not 200x; the biases held head-major ([*, H, N, N] with 16-byte aligned rows) so the attention levers read them in place",
           "[of3t_paircache] installed", "trunk_kernels/README.md:14; of3t_hook/sitecustomize.py:32,79; of3t_hook/of3t_paircache.py", T_MODEL),
    Lever("trimul_v4", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_PAIR",),
           "the pair stack's two TriMul statements (PairBlock.tri_mul_out_in) on the core's fused TriMul cell (opt_core.kernels.fpf_trimul_v4: LayerNorm + gated dual "
           "projection + contraction + LayerNorm-out + gate/projection + residual per direction; c_z = c_hidden in {128, 256}, N in the cell table's range, launch "
           "launch cells from the core's fpf_trimul_v4 table) — class-wide on PairBlock (trunk, MSA-module, confidence stacks); the template stack "
           "(c = 64) and out-of-range N run the runner yaml's own TriMul kernel by name (census fallback:<reason>=<n>; degraded=<n> counts trunk-shape refusals and "
           "OPENFOLD3_OB0_OPT_PAIR_STRICT=1 fails the run on one)",
           "[openfold3_ob0-opt/pairfused] installed", "opt/openfold3_ob0_opt/cells/pairfused.py; opt/openfold3_ob0_opt/hooks/cells/sitecustomize.py", T_MODEL),
    Lever("trimul_provider", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_TRIMUL_PROVIDER",),
           "the TriMul cell's row per call class (GPU, precision, c_z, c_hidden, token bucket, direction) chosen by the core's ONE triangle-multiplication provider (opt_core.kernels.trimul: every carried implementation a named row — v4 / tmk3_fast / tmk3_exact / bz2 / tx_sm90a / esm_v5_fwd / esm_v61 / cueq … — and the measured cell table TRIMUL_CELLS.json), word fast in this kit's measured row order (the line's tier word for every class): H100 — the kit row `v4` (this cell's own fpf_trimul_v4 statement, byte for byte) at the pairformer / MSA-module / confidence width (128,128) at every size (the cell's fast row on this stack), the provider's `tmk3_fast` (then `tmk3_exact`) at the template pair stack's (64,64), which the kit row does not serve; A100 — the kit row at (128,128), the template width on the line. A row the provider refuses for a class is counted by its word and the next asked; the caller's OPENFOLD3_OB0_OPT_TRIMUL_WORD names any provider word for an ablation; left off (MODEL_OPT_LEVERS_OFF=trimul_provider) the cell serves v4 / the line as before. Census `LEVER name=trimul_provider word=… rows=<row>:<n> kit=v4:<n> line=… skip=… witness=… cells=…`; tolerance class (the first statement per class and row is witnessed against the class's reference, a non-finite or gross one refused by name)",
           "[openfold3_ob0-opt/pairfused] trimul_provider:", "opt/openfold3_ob0_opt/cells/pairfused.py; opt/openfold3_ob0_opt/of3_trimul.py", T_MODEL),
    Lever("triatt_block", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_PAIR", "OPENFOLD3_OB0_OPT_PAIR_IMPL", "OPENFOLD3_OB0_OPT_PAIR_CORE"),
           "the pair stack's starting + ending triangle attention (PairBlock.tri_att_start_end) as the core's fused block (opt_core.attn.pair_fused.tri_attn_block: "
           "LayerNorm -> q|k|v|g + triangle-bias prologue, the core's flash triangle attention, sigmoid-gate * o @ W_o epilogue adding its output into z in place (a transposed scatter for the ending node); the ending node reads z^T by address "
           "math and projects its bias from z's own frame as openfold3 >= 0.5.0 does, bias_frame='z'); bf16 activations fully fused; an fp32 pair stream (the "
           "confidence head's pairformer_dtype=float32 stack) and shapes outside the core's cell table run the line's own tri-attention by name (fallback:stream:float32, "
           "fallback:<reason>)",
           "[openfold3_ob0-opt/pairfused] installed", "opt/openfold3_ob0_opt/cells/pairfused.py", T_MODEL),
    Lever("triatt_provider", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_PAIR_CORE",),
           "the fused triangle-attention block's attention core chosen per call class (GPU, dtype, head dim, heads, key count, call form) by the core's ONE triangle-attention provider (opt_core.kernels.triattn, cell table TRIATTN_CELLS.json) on the line's TIER word `fast` — the cell's measured fast-class row per class on each card (H100: the sealed sm_90a extension / the sealed triangle-attention package at head dim 32, the package or k2b at head dim 16; A100: the package's sm_80 member; k2b / k2 where a package has no prebuilt for the torch ABI), no kit-side row order; a row that cannot serve is refused by name and the cell's named fallback serves, counted; left off, the block's flash core (the block before this lever)",
           "[openfold3_ob0-opt/pairfused] installed", "opt/openfold3_ob0_opt/cells/pairfused.py; opt/openfold3_ob0_opt/of3_triattn.py", T_MODEL),
    Lever("pair_transition", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_PAIR",),
           "the pair stack's SwiGLU transition as the core's fused transition (opt_core.attn.pair_fused.transition: LayerNorm -> W_a|W_b -> silu(a)*b -> W_out, the "
           "4*c_z hidden never in HBM; the stock mask applied to the update); PairBlock's transition instances only, bf16 activations (an fp32 stream runs the "
           "stock transition by name: fallback:stream:float32)",
           "[openfold3_ob0-opt/pairfused] installed", "opt/openfold3_ob0_opt/cells/pairfused.py", T_MODEL),
    Lever("rollout_bf16", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_ROLLOUT", "OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS"),
           "the diffusion denoiser step — DiffusionModule (conditioning, the 24-block diffusion transformer, the token-level projections), 200 steps over the sample batch, "
           "and the graphed step's eager per-rollout work (the pair cache's token-pair buffers) — under torch.autocast(cuda, bfloat16) instead of upstream's fp32 rollout for "
           "every prediction (OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS, default 0, gates predictions below a caller's threshold: upstream's fp32 rollout runs throughout and the "
           "census says gated=<n> — a gate by design, not a fallback), with DiffusionModule.atom_attn_enc / .atom_attn_dec (the modules that read and write atom coordinates) under "
           "torch.autocast(cuda, float32) and the sampler's rotation / noise / update arithmetic left in upstream's fp32 (bf16's 8-bit mantissa on a coordinate is 0.06-0.25 A "
           "of bond-length noise): matmul-class ops bf16 with fp32 accumulation, LayerNorm / softmax statistics fp32 (autocast's op lists), the attention core stays fp32 "
           "where upstream asks use_high_precision_attention; holds inside the CUDA-graphed step (graphed outputs run-to-run bitwise; they differ from the eager bf16 sampler by the tier's own class)",
           "[openfold3_ob0-opt/rollout] installed", "opt/openfold3_ob0_opt/cells/rollout.py", T_MODEL),
    Lever("dit_attn", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_DIT", "OPENFOLD3_OB0_OPT_DIT_MIN_TOKENS", "OPENFOLD3_OB0_OPT_DIT_CORE"),
           "the diffusion transformer's attention with pair bias (24 blocks x 200 steps, DiffusionAttentionPairBias) on the core's flash kernel "
           "(opt_core.kernels.dtk_kernels.flash_bias_attn: the [S,H,N,N] logits never materialised; fp32 softmax statistics; all samples in one launch of the core's "
           "pair-bias attention — the core's ONE provider opt_core.kernels.apb asked by the family's word (cells/apb_word.py: the line's tier word OPENFOLD3_OB0_OPT_APB_TIER=fast|big, "
           "or the caller's OPENFOLD3_OB0_OPT_APB_WORD), the direct entry opt_core.attn.apb_core when no word is set; OPENFOLD3_OB0_OPT_DIT_CORE=dtk asks the provider's dtk_loop row —, the shared pair bias read "
           "in place from the pair cache's head-major buffer) inside the sampler's CUDA graph, on 16-bit operands only — the bf16 roll-out's (rollout_bf16, from 600 "
           "tokens by that lever's gate), where upstream's use_high_precision_attention word (fp32 attention core) is overridden and counted (census "
           "high_precision_overridden=<n>); fp32 operands (the fp32 roll-out) run upstream's core by design, gated by dtype (gated_by dtype_fp32=<n>), and a size gate "
           "(opt_core.attn.size_gate, min tokens OPENFOLD3_OB0_OPT_DIT_MIN_TOKENS default 256) gates small inputs likewise (gated_by lt_min=<n>); other shapes = named "
           "fallback. In the fast line (OPENFOLD3_OB0_OPT_DIT=flash_bias_attn)",
           "[openfold3_ob0-opt/dit_attn] installed", "opt/openfold3_ob0_opt/cells/dit_attn.py; opt/openfold3_ob0_opt/hooks/cells/sitecustomize.py", T_MODEL),
    Lever("dit_glue", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_DIT_GLUE", "OPENFOLD3_OB0_OPT_DIT_GLUE_MIN_TOKENS", "OPENFOLD3_OB0_OPT_DIT_GLUE_CORE", "OPENFOLD3_OB0_OPT_DIT_GLUE_ROWS"),
           "the token diffusion transformer block (DiffusionTransformerBlock.forward: 24 blocks x 200 steps over the samples) as ONE schedule of fused pieces instead of "
           "~130 kernels: AdaLN on the core's ln_modulate with the conditioning rows periodic over the samples (never expanded), ONE q|k|v|gate GEMM, the pair-bias "
           "attention on its four column slices in place (the core's pair_bias_attention entry when carried, else dtk_kernels.flash_bias_attn; the pair cache's bias read "
           "in place), ONE output GEMM, the adaLN-zero gate + residual in one row kernel, the conditioned SwiGLU transition as one [a|b] GEMM + swiglu + GEMM + "
           "gate/mask/residual kernel; the conditioned transition and AdaLN OUTSIDE that schedule (the atom transformer's blocks, fp32, and any refused token "
           "block) served class-wide by the same row schedules (OPENFOLD3_OB0_OPT_DIT_GLUE_ROWS=all|block, default all) (opt_core.of3_sampler.dit_glue + "
           "dit_rows, bound by cells/dit_glue.py); on the bf16 roll-out's 16-bit operands (the fp32 roll-out "
           "keeps the stock block: gated:cd_float32), upstream's use_high_precision_attention word overridden and counted (high_precision_overridden=<n>) as dit_attn "
           "does; refuses by name -> the stock block, counted. In the fast line (OPENFOLD3_OB0_OPT_DIT_GLUE=1)",
           "[openfold3_ob0-opt/dit_glue] installed", "opt/openfold3_ob0_opt/cells/dit_glue.py", T_MODEL),
    Lever("apb_trunk", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_APB_TRUNK", "OPENFOLD3_OB0_OPT_APB_TRUNK_MIN_TOKENS", "OPENFOLD3_OB0_OPT_APB_TRUNK_HIGH_PRECISION", "OPENFOLD3_OB0_OPT_APB_TRUNK_SCOPE", "OPENFOLD3_OB0_OPT_APB_TRUNK_PRODUCER"),
           "the pairformer single track's attention with pair bias (AttentionPairBias of the 48-block trunk per recycle pass and of the confidence head's "
           "pairformer per sample; the diffusion transformer's DiffusionAttentionPairBias is dit_attn's / dit_glue's): the pair bias produced head-major (the module's "
           "LayerNorm(z), then one W_z @ LN(z)^T GEMM writing the [H,N,N] planes directly: no [N,N,H] intermediate, no permute pass, no per-call relayout) and the "
           "attention as ONE launch of the core's pair-bias kernel (opt_core.attn.apb_core: the strided projection views in place, the key mask by name, the sigmoid "
           "gate fused, the [H,N,N] logits never materialised); upstream's use_high_precision_attention word (asked on every trunk call: fp32 logits, softmax and "
           "P.V) is overridden on the bf16 trunk's 16-bit operands (fp32 logit accumulation and softmax statistics) and counted (high_precision_overridden=<n>) as "
           "dit_attn / dit_glue do; the pair-bias producer is switchable (OPENFOLD3_OB0_OPT_APB_TRUNK_PRODUCER=ln_proj|mm: the core's fused LayerNorm+projection "
           "kernel opt_core.kernels.ln_proj, the default, or the module's LayerNorm + one GEMM; another lever may register its own through use_producer); "
           "refuses by name -> the stock forward, counted (opt_core.of3_sampler.apb_trunk, bound by cells/apb_trunk.py). Attribution knobs: OPENFOLD3_OB0_OPT_APB_TRUNK_HIGH_PRECISION=override|honour "
           "(default override), OPENFOLD3_OB0_OPT_APB_TRUNK_SCOPE=trunk+confidence|trunk (default trunk+confidence; `trunk` runs the confidence head's calls on the stock forward, counted "
           "scope:confidence). In the fast line (OPENFOLD3_OB0_OPT_APB_TRUNK=1)",
           "[openfold3_ob0-opt/apb_trunk] installed", "opt/openfold3_ob0_opt/cells/apb_trunk.py", T_MODEL),
    Lever("token_agg", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_TOKEN_AGG",),
           "the atom -> token mean of the atom-attention encoder (atomize_utils.aggregate_atom_feat_to_tokens: a full-shape int64 index + a float-atomic scatter-add per "
           "denoiser step) as ONE deterministic segment-reduce kernel over each token's contiguous atom run (dtk_kernels.seg_reduce; the runs derived once per rollout, "
           "refreshed in place: opt_core.of3_sampler.rollout_memo); batch-1 inference, any sample / token / atom count; other layouts refused by name -> stock. In the fast "
           "line (OPENFOLD3_OB0_OPT_TOKEN_AGG=seg_reduce)",
           "[openfold3_ob0-opt/token_agg] installed", "opt/openfold3_ob0_opt/cells/token_agg.py", T_MODEL),
    Lever("atom_window", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_ATOM_WINDOW", "OPENFOLD3_OB0_OPT_ATOM_WINDOW_PRECISION", "OPENFOLD3_OB0_OPT_ATOM_WINDOW_INV"),
           "the diffusion sampler's sequence-local atom attention (AtomAttentionEncoder / Decoder: 3 + 3 DiffusionTransformerBlocks x 200 steps over "
           "the samples) on the core's two fused kernels (opt_core.kernels.atom_window: LayerNorm + both AdaLN modulations from per-atom conditioning "
           "rows + the q|k|v|gate projections in one pass with k / v once per atom instead of on gathered 128-key window copies; then, per query block "
           "and sample, the shifted key window read in place, pair bias + validity mask, fp32 softmax, p v, sigmoid gate, W_o, AdaLN-Zero gate and "
           "residual in one kernel — TF32 round-to-nearest dots, fp32 in/out, on both sides of rollout_bf16's token gate); the step-invariant "
           "conditioning / bias / mask operands once per rollout (opt_core.of3_sampler.rollout_memo); upstream's use_high_precision_attention word is served as "
           "asked (fp32 logits and softmax); refuses by name -> the stock block, counted (opt_core.of3_sampler.atom_window, bound by cells/atom_window.py)",
           "[openfold3_ob0-opt/atom_window] installed", "opt/openfold3_ob0_opt/cells/atom_window.py", T_MODEL),
    Lever("atom_hoist", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_ATOM_HOIST", "OPENFOLD3_OB0_OPT_ATOM_HOIST_MAX_GB", "OPENFOLD3_OB0_OPT_ATOM_HOIST_INV"),
           "the atom path's step-invariant work once per rollout instead of per denoiser step: AtomAttentionEncoder.get_atom_reps (c_l, p_lm: reference embedding, pair "
           "activations incl. LN+linear of the trunk pair, pair MLP), the atom transformers' LN_z(p_lm) and per-block pair-bias linear_z, the block-index utilities — "
           "memoised per rollout in address-stable buffers refreshed in place (opt_core.of3_sampler.rollout_memo; under the CUDA-graphed step the first denoiser call of each "
           "rollout runs eager to refresh them); every stock expression kept: bitwise the line without it; publishes per block the `_of3opt_atom_inv` buffers. In the "
           "exact and fast lines (OPENFOLD3_OB0_OPT_ATOM_HOIST=1)",
           "[openfold3_ob0-opt/atom_hoist] installed", "opt/openfold3_ob0_opt/cells/atom_hoist.py", T_MODEL),
    Lever("templ_embed", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_TEMPL_EMBED",),
           "the template embedder around the template pair stack (TemplateEmbedderAllAtom, once per trunk pass over T templates x N x N): the eight bias-free "
           "feature Linears + linear_z(LayerNorm(z)) as ONE kernel writing the stack's [T,N,N,64] input (fp32 accumulation, one bf16 rounding) and the "
           "mean over templates / relu / linear_t as ONE kernel (a stride-0 template axis read in place) — opt_core.of3_trunk.templ_embed on the core's "
           "templ_embed kernels, bound by cells/templ_embed.py with OpenFold3 0.5.0's forward signature (the resident path served; upstream's offloaded "
           "per-template path, offload_inference=True above its token cutoff, is upstream's statement counted by name); the stack itself untouched; bf16 "
           "autocast, batch 1, any template count / token count / chain layout; refuses by name -> the stock forward, counted. In the fast line "
           "(OPENFOLD3_OB0_OPT_TEMPL_EMBED=1)",
           "[openfold3_ob0-opt/templ_embed] installed", "opt/openfold3_ob0_opt/cells/templ_embed.py", T_MODEL),
    Lever("castcache", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_CASTCACHE",),
           "the engine's Linear primitive casts its fp32 weight and bias to bf16 on every bf16 call (two copy kernels per call); the bf16 copies memoised per module "
           "(key: the fp32 tensors' storage + version — reloaded, moved or edited weights refresh it) and handed to the same F.linear call: bitwise the line without "
           "it (opt_core.of3_trunk.castcache; this engine's LayerNorm upcasts bf16 input to fp32 instead of casting its weight — that primitive runs as it is, named "
           "`skipped=layernorm` on the census). In the exact and fast lines (OPENFOLD3_OB0_OPT_CASTCACHE=1)",
           "[openfold3_ob0-opt/castcache] installed", "opt/openfold3_ob0_opt/cells/castcache.py", T_MODEL),
    Lever("exactln", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_EXACTLN", "OPENFOLD3_OB0_OPT_EXACTLN_WORD"),
          "the `LayerNorm` primitive (the one class behind every LayerNorm of the trunk, the MSA / template stacks, the diffusion transformer and the heads; the upcast dialect: "
          "fp32 statistics, the result cast back) bound to the shared core's LayerNorm provider (opt_core.kernels.ln: one provider over the NVRTC replica of ATen's kernel `exactln`, the "
          "Triton rows and the measured cell table) by the word `exact`: per call class (form, width, row count, token bucket, eager | graph, stack) the provider names an exact-class "
          "row only where it is measured bitwise F.layer_norm AND at least as fast as ATen on the running card and stack, else ATen by name (the statement runs, counted); every "
          "exact-class row's first call per signature proven torch.equal against the statement in-process (a differing signature retires to the statement, named); the replica's kernels "
          "compile and self-check in a background thread at install; other dtypes / devices / non-contiguous / autocast-enabled calls keep the statement, counted. In the exact line "
          "(cells/exactln.py, of3_exactln.py; knob OPENFOLD3_OB0_OPT_EXACTLN_WORD=<tier word | provider row>)",
          "[openfold3_ob0-opt/exactln] installed", "opt/openfold3_ob0_opt/cells/exactln.py", T_MODEL),
    Lever("ln_provider", "cells", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_LN_PROVIDER", "OPENFOLD3_OB0_OPT_LN_TIER", "OPENFOLD3_OB0_OPT_LN_WORD"),
          "the same LayerNorm binding asked with the line's TIER word — `fast` on the fast line (the provider's fastest measured row inside each cell's identity band: the Triton rows "
          "`fastln:lp` / `fastln` / `ln_rows` on the N²-row pair and S·N-row MSA norms, the replica or ATen on the N-row operands, capture-aware — a call met during CUDA-graph capture asks "
          "the graph cell), `big` on the memory line (the lowest-peak row among those); a tolerance row's first call per signature witnessed against the statement (rel-RMS bound, else "
          "the signature retires to the statement by name), exact-class rows proven bitwise; cells naming ATen run the statement, counted. In fast and big/resident "
          "(cells/ln_provider.py, of3_exactln.py; knob OPENFOLD3_OB0_OPT_LN_WORD=<tier word | provider row>)",
          "[openfold3_ob0-opt/ln_provider] installed", "opt/openfold3_ob0_opt/cells/ln_provider.py", T_MODEL),
    Lever("triatt_exact", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_TRIATT_EXACT",),
           "the cuEquivariance triangle-attention call of the exact line's pair stacks (primitives.attention.triangle_attention, every call above the library's 100-token threshold) handed to the core's triangle-attention provider on word exact: at this pin the cell names the stock library call on every card and it serves, by name; an exact-class row a cell names would serve a call class only after its first eager call proved torch.equal against the library on the call's own operands (a differing class is refused for the process by name): bitwise the line without it (openfold3_ob0_opt.of3_triattn)",
           "[openfold3_ob0-opt/triattn] triatt_exact: installed", "opt/openfold3_ob0_opt/cells/triatt_exact.py; opt/openfold3_ob0_opt/of3_triattn.py", T_MODEL),
    Lever("trimul_exact", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_TRIMUL_EXACT",),
           "the pair stacks' triangle multiplicative updates (outgoing + incoming) served by the core's ONE triangle-multiplication provider (opt_core.kernels.trimul, cell table TRIMUL_CELLS.json, binding opt_core.trimul.by_word) on word exact: per call class the exact-class row the table measured bitwise and at or above the line's op — tmk3_exact (the cuEquivariance TriMul's operations in order on the core's kernels) at c_z 128 up to 400 tokens and at c_z 64 (template pair stack) at every size, H100 and A100; the line's own cuEquivariance call everywhere else, by name; a kernel row serves a class only after its first eager call proved torch.equal against the line's statement (a class whose bits differ is refused for the process, named)",
           "[openfold3_ob0-opt/trimul_exact] installed", "opt/openfold3_ob0_opt/cells/trimul_exact.py", T_MODEL),
    Lever("trimul_form", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_TRIMUL_EXACT_FORM",),
           "rides on trimul_exact: the TriMul call classes whose statement is the MODULE's own inference path (use_cueq_triangle_kernels False at the call: the template pair stack, c_z 64, on every input; every pair block when the runner configuration switches the library off) served by the core's trimul provider asked for the exact tier under this engine's form key (opt_core.kernels.trimul select(word=exact, form=of3_module) -> the row of3_form where a form cell proves it, refused by name elsewhere; >= 0.5.67.0: that statement issued whole-tensor -- the projections and the gate as the module's GEMMs once over the pair tensor, the contraction in upstream's 256-column blocks, LayerNorms on the core's exactln row) once the first call of each (class, token count) proved torch.equal against the module on that call's operands (a shape whose bits differ refuses its class for the process, named); left off, those classes run the module's statement; H100-tested",
           "[openfold3_ob0-opt/trimul_form] armed", "opt/openfold3_ob0_opt/cells/trimul_form.py; opt/openfold3_ob0_opt/cells/trimul_exact.py", T_MODEL),
    Lever("transition_exact", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_TRANSITION_EXACT",),
           "the exact line's SwiGLU transitions (layers.transition.SwiGLUTransition: the pairformer, MSA-module and confidence pair-stack transitions, c 128 / hidden 512; the template pair stack's c 64) bound to the core's ONE transition provider (opt_core kernels.transition) by the tier word exact under this engine's statement form (form swiglu), fed the module's own LayerNorm output: where the provider's table names a kernel row (v1: everything after the LayerNorm one Triton kernel) the exact-tier winner for the cell on this process's stack - bitwise against the statement there, from the stack's token floor up - the call launches it; a stock winner, a cell without a vouch on this stack or row count, no cell, the c 384 single transition, fp32 calls and module variants run the module's statement, counted with the provider's word; no kernel, tile table or floor in the kit: bitwise the line without it (openfold3_ob0_opt.of3_transition)",
           "[openfold3_ob0-opt/transition_exact] installed", "opt/openfold3_ob0_opt/cells/transition_exact.py; opt/openfold3_ob0_opt/of3_transition.py", T_MODEL),
    Lever("apb_hoist", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_APB_HOIST",),
           "AttentionPairBias._prep_bias' key-mask bias `inf * (mask - 1)` memoised per stack call (the mask is constant over the 48 pairformer blocks of a recycle "
           "pass) instead of rebuilt per block; the pair-bias term computed as stock: bitwise (opt_core.of3_trunk.apb_hoist; on the fast line apb_trunk serves the "
           "pairformer / confidence instances and passes the rest here). In the exact and fast lines (OPENFOLD3_OB0_OPT_APB_HOIST=1)",
           "[openfold3_ob0-opt/apb_hoist] installed", "opt/openfold3_ob0_opt/cells/apb_hoist.py", T_MODEL),
    Lever("trunk_graph", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_TRUNK_GRAPH", "OPENFOLD3_OB0_OPT_TRUNK_GRAPH_NMAX"),
           "PairFormerStack.forward (the 48-block trunk stack per recycle pass, the confidence head's 4-block stack per call; launch-bound below ~1000 tokens) captured "
           "into a CUDA graph per (instance, input signature, flags, numerics mode) at its second call and replayed after, first call eager, one process pool; token "
           "count above OPENFOLD3_OB0_OPT_TRUNK_GRAPH_NMAX (1024), grad, CPU, nested capture, DS4Sci attention, lma or a failed capture run the stack's own forward by "
           "name: bitwise (opt_core.of3_trunk.trunk_graph). In the fast line (OPENFOLD3_OB0_OPT_TRUNK_GRAPH=1)",
           "[openfold3_ob0-opt/trunk_graph] installed", "opt/openfold3_ob0_opt/cells/trunk_graph.py", T_MODEL),
    Lever("post_release", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_POST_RELEASE", "OPENFOLD3_OB0_OPT_POST_RELEASE_NTOK", "OPENFOLD3_OB0_OPT_POST_RELEASE_MIN_FREE_GB"),
           "the graphed lines' item-boundary memory release at large token counts (>= OPENFOLD3_OB0_OPT_POST_RELEASE_NTOK, 2400, or free memory < "
           "OPENFOLD3_OB0_OPT_POST_RELEASE_MIN_FREE_GB, 16): the graphed step's generations + pool + gather tables, the pair cache's static buffers, trunk_graph's graphs, "
           "cuBLAS workspaces and cached-free allocator segments released right after the sampler returns (before the confidence head's per-sample pair batch) and "
           "after the item's forward; MiB freed per source on the census; the next item re-captures; outputs untouched: bitwise (opt_core.of3_sampler.post_release). "
           "In the exact and fast lines (OPENFOLD3_OB0_OPT_POST_RELEASE=1)",
           "[openfold3_ob0-opt/post_release] installed", "opt/openfold3_ob0_opt/cells/post_release.py", T_MODEL),
    Lever("tuner_guard", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_TUNER_GUARD",),
           "the engine's per-stack chunk-size tuner compares the argument record of its last tuning with the next chunked call's and raises when "
           "they differ in structure (the confidence stack's batched call at or below its 750-token per-sample cutoff vs its per-sample calls above: "
           "tensor ranks differ) — one process predicting a query of <= 750 tokens and then a larger one failed every larger query on the kernels-off "
           "runner configuration; the kit answers such a comparison 'changed' so the tuner re-tunes, counted (opt_core.of3_trunk.tuner_guard)",
           "[openfold3_ob0-opt/tuner_guard] installed", "opt/openfold3_ob0_opt/cells/tuner_guard.py", T_MODEL),
    Lever("postfwd_mem", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_POSTFWD_MEM", "OPENFOLD3_OB0_OPT_POSTFWD_MEM_MIB"),
           "the runner's confidence scoring after the forward (the engine's get_confidence_scores on the forward's outputs) without the "
           "[S, N, N, 64] probability tensors (pde, pae, every compute_ptm call) and the [S, N_atom, N_atom, 3] all-atom difference tensor of "
           "get_token_frame_atoms that set the process peak of every line after OpenFold3.forward returned: those statement groups run per "
           "(sample, row block) under a MiB budget (_MIB, default 256; 0 = the engine's statements) and the engine's full-size result tensors are "
           "assembled by copy, every long reduction on the engine's own tensors — bitwise; on big/resident it serves below the confidence gate "
           "(modes.CONF_MIN_TOKENS), the port's chunked scorer above it (openfold3_ob0_opt.of3_postfwd)",
           "[openfold3_ob0-opt/postfwd_mem] installed", "opt/openfold3_ob0_opt/cells/postfwd_mem.py; opt/openfold3_ob0_opt/of3_postfwd.py", T_MODEL),
    Lever("loader_workers", "cells", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_LOADER_WORKERS", "OPENFOLD3_OB0_OPT_LOADER_WORKERS_CAP"),
           "the engine's predict DataLoader forks data_module_args.num_workers (10) featurisation workers whatever the query set holds; the kit "
           "caps the predict loader at max(1, min(items, configured)) — the same worker featurises the same item with the same seed (in-order "
           "loader, per-worker seeding by index): byte-identical outputs, fewer forked interpreters; _CAP=none keeps the engine's count "
           "(openfold3_ob0_opt.of3_loader)",
           "[openfold3_ob0-opt/loader_workers] installed", "opt/openfold3_ob0_opt/cells/loader_workers.py; opt/openfold3_ob0_opt/of3_loader.py", T_MODEL),
    Lever("conf_dtype", "package", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_CONF_DTYPE",),
           "the confidence phase (AuxiliaryHeadsAllAtom.forward: the 4-block confidence Pairformer, the PAE / PDE / pLDDT / resolved heads, the distogram head) under "
           "torch.autocast(cuda, bfloat16) with pairformer_dtype=bfloat16 — openfold3 0.4.1's dtype of the phase — instead of upstream 0.5.0's inference fp32 (model.py "
           "cast_dtype = si_trunk.dtype after run_trunk's .float(); head_modules pairformer_dtype=float32); the word OPENFOLD3_OB0_OPT_CONF_DTYPE=bf16|fp32 is the line's "
           "(bf16 on fast / big, fp32 on exact) unless preset or `--conf-dtype` (source=env); the trunk outputs and the diffusion sampler keep their dtypes, the phase runs "
           "after the sampler: coordinates bitwise those of fp32 heads, confidence numbers tier 2; the row-sharded line's own statement of the phase reads the same word",
           "[openfold3_ob0-opt/conf_dtype] installed", "opt/openfold3_ob0_opt/conf_dtype.py: installed by the package's activation (stack.ACTIVATION_MODULES) in every active process", T_HEAD_MODULES),
    Lever("z_dtype", "package", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_Z_DTYPE",),
          "the trunk's hand-off to the roll-out (run_trunk's s_input, s, z) in the trunk's own bfloat16 as OpenFold3 0.4.1 hands it (`return s_input, s, z`) "
          "instead of upstream 0.5.0's fp32 copies (`return s_input.float(), s.float(), z.float()`, model.py:325): the diffusion conditioning, the confidence "
          "z-embedding / Pairformer and the distogram head then read a bf16 pair representation (N²·256 B instead of N²·512 B; on the row-sharded line per rank, and "
          "its diffusion-stage pair caches follow); statements: OpenFold3.run_trunk wrapped at import (fast: `.to(bfloat16)` after upstream's `.float()`, exact values), "
          "the resident port's and the row-sharded trunk's own hand-off `z_dtype.handoff` (no copy); the word OPENFOLD3_OB0_OPT_Z_DTYPE=bf16|fp32 is the line's (bf16 on "
          "fast / big, fp32 on exact) unless preset or `--z-dtype` (source=env); COORDINATES MOVE (the sampler reads z): tier 2, evidence in CHANGES.md; "
          "`z_dtype=bf16` with `conf_dtype=fp32` is refused (upstream's cast_dtype follows the hand-off)",
          "[openfold3_ob0-opt/z_dtype] installed:", "opt/openfold3_ob0_opt/z_dtype.py: installed by the package's activation (stack.ACTIVATION_MODULES) in every active process", T_MODEL),
    Lever("fastjson", "package", OUTPUT, EXACT, ("OPENFOLD3_OB0_OPT_FASTJSON", "OPENFOLD3_OB0_OPT_FASTJSON_ROUTE"),
           "the engine's output writer renders each sample's aggregated confidence JSON and, under the runner's json full-confidence format, the full "
           "confidence JSON (plddt per atom, pde / pae per token pair; json.dumps(indent=4, cls=NumpyEncoder): CPython's pure-Python indent encoder) through "
           "a subclass of its encoder class whose encode() builds the same characters row by row at C level (str.join over float.__repr__, NaN / Infinity as "
           "json spells them, the engine's own rounded array conversion) — every output file identical to the unpatched writer's; the model forward untouched, "
           "the item's wall time after it shorter; a structure the port does not reproduce is the stock encoder's call, counted; kept only when its probe text "
           "equals the stock class's at install (cells/of3_fastjson.py; OPENFOLD3_OB0_OPT_FASTJSON_ROUTE=stock: the class installed, every call the stock encoder's)",
           "[openfold3_ob0-opt/fastjson] installed", "opt/openfold3_ob0_opt/fastjson.py: installed by the package's activation (stack.ACTIVATION_MODULES) in every active process", T_WRITER),
    Lever("writer_overlap", "package", OUTPUT, EXACT, ("OPENFOLD3_OB0_OPT_WRITER_OVERLAP", "OPENFOLD3_OB0_OPT_WRITER_OVERLAP_ROUTE"),
           "the engine's output writer callback (OF3OutputWriter.on_predict_batch_end -> write_all_outputs: per diffusion sample one mmCIF through "
           "biotite and two indented JSON texts, on the main thread between two forwards) re-stated so the callback's bookkeeping and the "
           "device-to-host copies the writer makes first stay where the engine runs them and the rendering + writing — the writer instance's own "
           "write_all_outputs on those host copies, fastjson applying unchanged — runs in one writer worker (a forked child process by default: no "
           "interpreter lock shared with the next forward's launches; a thread or inline by knob) while item k+1 is featurised and forwarded, at most "
           "two items in flight, every item drained and tallied into the callback's own counters before the engine's summary and at exit; "
           "write_features / write_latent_outputs / a distributed predict written inline by name — every output file identical, the wall time "
           "between two forwards shorter, the model forward untouched (cells/of3_writer_overlap.py; OPENFOLD3_OB0_OPT_WRITER_OVERLAP_ROUTE=process|thread|sync)",
           "[openfold3_ob0-opt/writer_overlap] installed", "opt/openfold3_ob0_opt/writer_overlap.py: installed by the package's activation (stack.ACTIVATION_MODULES) in every active process", T_WRITER),
    Lever("hostfeat", "package", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_HOSTFEAT", "OPENFOLD3_OB0_OPT_HOSTFEAT_PARTS"),
           "the a3m deletion matrix + aligned letter matrix of the engine's MSA reader (core.data.io.sequence.msa.parse_a3m: a per-character Python "
           "loop over every MSA row, then translate + a row-by-row '<U1' fill) computed from the byte view of the rows, then the engine's own tail "
           "(MsaArray.from_parsed + truncate) — integer / string arrays only, every feature identical (the kit tests against the engine's text); "
           "item 1's featurisation, which every process's first forward waits for, seconds shorter at the ladder sizes; a row outside the byte "
           "identity (non-ASCII, ragged) or an engine text the port does not re-state goes to the engine's statement by name; the MSA letter table "
           "part (msaidx) is upstream's own at 0.5.0 and not bound here (cells/of3_hostfeat.py; OPENFOLD3_OB0_OPT_HOSTFEAT_PARTS=a3m)",
           "[openfold3_ob0-opt/hostfeat] installed", "opt/openfold3_ob0_opt/hostfeat.py: installed by the package's activation (stack.ACTIVATION_MODULES) in every active process", T_MSA_IO),
    Lever("ckpt_mmap", "package", FORWARD, EXACT, ("OPENFOLD3_OB0_OPT_CKPT_MMAP",),
           "the engine's checkpoint read (core.utils.checkpoint_loading_utils.load_checkpoint: torch.load of 2.3 GB of fp32 tensors, unpickled "
           "into fresh host memory in every process before the model can load them) made through the same torch.load call with mmap=True — "
           "the storages memory-mapped instead of copied, the same dict of the same tensors handed to get_state_dict_from_checkpoint / "
           "load_state_dict; a directory checkpoint or a file torch cannot map is the engine's own call, counted (cells/of3_ckptmmap.py)",
           "[openfold3_ob0-opt/ckpt_mmap] installed", "opt/openfold3_ob0_opt/ckpt_mmap.py: installed by the package's activation (stack.ACTIVATION_MODULES) in every active process", T_CKPT),
    Lever("tp_shard_s", "tp_rowpair", FORWARD, TOLERANCE, (),
           "tensor parallelism: the pair representation row-sharded across the ranks (one process per GPU, an NCCL group) — the trunk's pair stack, "
           "MSA module, template embedder, diffusion conditioning and confidence heads run on each rank's rows with ring-streamed / transposed peers "
           "(opt_core.mem.rowpair bound by opt/openfold3_ob0_opt/tp_rowpair); the LayerNorm 2^31-element guard; not bitwise vs the one-card routes on this tree",
           "[tp_rowpair hook r", "opt/openfold3_ob0_opt/tp_rowpair/__init__.py; opt/openfold3_ob0_opt/tp_rowpair/hook/sitecustomize.py; opt/openfold3_ob0_opt/tp_rowpair/env.py; opt/openfold3_ob0_opt/tp.py", T_MODEL),
    Lever("tp_triatt", "tp_rowpair", FORWARD, TOLERANCE, (),                       # on with the tp line: the flash_triattn kernel word (tp_rowpair/pairstack.TRIATT_KERNEL); the core's opt-out ROWPAIR_TRIATT_CORE=torch
           "the row-sharded pair stack's triangle attention on the core's row-block kernel dispatch (opt_core.mem.rowpair.triatt.attention_core kernel=<word>, ONE "
           "vocabulary): flash_triattn = the core's carried flash triangle-attention kernel per row batch; OpenFold3's "
           "torch attention statement is the dispatch's named per-call fallback (the core's LEVER name=F1.flash_triattn line: served / fallback / fallback_by; one "
           "[tp_rowpair] TRIATT line per rank, the run record's tp block); a word the interpreter cannot serve is refused by name at bind. Not bitwise vs the torch statement",
           "[tp_rowpair] TRIATT kernel=", "opt/openfold3_ob0_opt/tp_rowpair/pairstack.py (triatt_kernel, triatt_fns over opt_core.mem.rowpair.triatt.attention_core); opt/openfold3_ob0_opt/tp.py (census)", T_MODEL),
    Lever("tp_trimul", "tp_rowpair", FORWARD, TOLERANCE, (),                       # on with the tp line: the fpf_v4 provider word (tp_rowpair/pairstack.TRIMUL_KERNELS); the core's opt-out ROWPAIR_TRIMUL_KERNELS=torch
           "the row-sharded pair stack's triangle multiplication on the core's fused row-block provider (opt_core.mem.rowpair.trimul_fused around OpenFold3's torch "
           "TriMulFns: the carried fpf_trimul_v4 kernels — K1 LayerNorm + one gated projection per launch written into the GEMM block, K3 LayerNorm-out + W_o + gate + "
           "residual per contraction tile; launch cells from the core's fpf_trimul_v4 table per (cc, triton); c_z = c_hidden in {128,256}; pairs below the provider's "
           "size gate (2048 tokens) and any unit the provider declines run the torch statements, counted by reason (the core's LEVER name=F2.trimul_rows line; one "
           "[tp_rowpair] TRIMUL line per rank, the run record's tp block). Not bitwise vs the torch statements",
           "[tp_rowpair] TRIMUL kernels=", "opt/openfold3_ob0_opt/tp_rowpair/pairstack.py (trimul_kernels, trimul_fns over opt_core.mem.rowpair.trimul_fused); opt/openfold3_ob0_opt/cells/pairfused.py (trimul_weights); opt/openfold3_ob0_opt/tp.py (census)", T_MODEL),
    Lever("sample_loop", "tp_rowpair", FORWARD, TOLERANCE, (),                     # on with the tp line, no switch of its own
          "the diffusion roll-out and the confidence head one sample chunk at a time (opt_core.mem.sample_loop over opt_core.mem.ckpt's stock-order draws: "
          "the generator rewound per pass, every batch-shaped draw made at the S shape and sliced, per-pass outputs assembled on the host in stock order) — "
          "device memory independent of --num-diffusion-samples; one sample per pass on the row-sharded line",
          "[sample_loop] samples=", "opt/openfold3_ob0_opt/sample_loop.py; opt/openfold3_ob0_opt/tp_rowpair/model.py (rollout_rows)", T_MODEL),
    # --- the offload port (opt/forward/offload; the `big` mode): the O1 (resident) memory levers. The numerics
    # class of each lever is the line's tier word (modes.LINE_TIER per line): the O1 levers restate stock at stock chunk boundaries and are
    # entered as exact
    # (the port's fail-closed carry: offload/README.md). Every lever's record is
    # of3_offload.census() (the marker: the add-on's `[of3o]` JSON events on stderr); the hook fires after the model module executes.
    Lever("trimul_hostsnap", "offload", FORWARD, EXACT, ("OF3O_TRIMUL", "OF3O_ROWS"),
           "O1: triangle multiplication with the residual snapshot on the host — the pair update streamed in row blocks (OF3O_ROWS) from a pinned host copy instead of a second GPU copy of z",
           "[of3o]", "offload/of3o/of3_offload.py (trimul_inference_forward); offload/README.md", T_MODEL),
    Lever("triatt_lean", "offload", FORWARD, EXACT, ("OF3O_TRIATT",),
           "O1: the ending triangle attention on a lean transposed copy (one transient N×N buffer, released before the transition)",
           "[of3o]", "offload/of3o/of3_offload.py:272-324", T_MODEL),
    Lever("trans_inplace", "offload", FORWARD, EXACT, ("OF3O_TRANS",),
           "O1: the pair transition applied in place in row blocks (no full-size intermediate)",
           "[of3o]", "offload/of3o/of3_offload.py:325-354", T_MODEL),
    Lever("cond_once", "offload", FORWARD, EXACT, ("OF3O_COND", "OF3O_COND_ROWS"),
           "O1: the diffusion conditioning's pair path computed once per item in row blocks (OF3O_COND_ROWS) and cached for every sample and step",
           "[of3o]", "offload/of3o/of3_offload.py:413-481", T_MODEL),
    Lever("input_rows", "offload", FORWARD, EXACT, ("OF3O_INPUT", "OF3O_INPUT_ROWS", "OF3O_EMBED_ROWS"),
           "O1: the input embedder's relative-position pair features built in row blocks (OF3O_INPUT_ROWS) and the pairformer embedding in row blocks (OF3O_EMBED_ROWS) — no N×N×features intermediate",
           "[of3o]", "offload/of3o/of3_offload.py:356-411,824-882", T_MODEL),
    Lever("recycle_rows", "offload", FORWARD, EXACT, ("OF3O_RUN_TRUNK", "OF3O_RECYCLE_ROWS"),
           "O1: the recycling embedder's pair add in row blocks from the host snapshot (OF3O_RECYCLE_ROWS) inside the port's run_trunk",
           "[of3o]", "offload/of3o/of3_offload.py:534-652", T_MODEL),
    Lever("templ_host", "offload", FORWARD, EXACT, ("OF3O_TEMPL", "OF3O_TEMPL_HOST", "OF3O_TEMPL_ROWS", "OF3O_TEMPL_EMBED_ROWS", "OF3O_TEMPL_DEDUP"),
           "O1: template pair features kept on the host and embedded in row blocks (OF3O_TEMPL_ROWS / OF3O_TEMPL_EMBED_ROWS), duplicate templates de-duplicated",
           "[of3o]", "offload/of3o/of3_offload.py:946-1102", T_MODEL),
    Lever("conf_chunked", "offload", FORWARD, EXACT, ("OF3O_CONF_MODE", "OF3O_CONF_ROWS", "OF3O_CONF_MIN", "OF3O_CONF_TM_BACKEND"),
           "the confidence scorer over row blocks: PAE/PDE/pLDDT/TM from the pair logits in rows (OF3O_CONF_ROWS) with the BLOCKREDUCE TM backend (explicit; the native backend is a counted event, never a silent switch) — the consumer the row-block head (`confhead`) and the host logits (`logits_host`) require: the big lines carry it with them from modes.CONF_MIN_TOKENS tokens up (below: `stock` = upstream's scorer); on its own it moves no peak",
           "[of3o]", "offload/of3o/of3_offload.py:915-976; offload/of3o/of3o_confidence.py; offload/of3o/of3o_blockreduce.py", T_MODEL),
    Lever("confhead", "confhead", FORWARD, TOLERANCE, ("OPENFOLD3_OB0_OPT_CONFHEAD",),
           "the PAE/PDE confidence heads evaluated per scorer row block (opt/openfold3_ob0_opt/confhead.py: the heads' forward returns a RowBlockLogits — the pair representation stays, the [S,N,N,64] logits never exist; the head's LayerNorm + C->64 GEMM run on the block the chunked scorer slices) — the resident line's confidence-phase peak from modes.CONF_MIN_TOKENS tokens up (modes.conf_gate; requires conf_chunked)",
           "[openfold3_ob0-opt/confhead] installed", "opt/openfold3_ob0_opt/confhead.py; opt/openfold3_ob0_opt/hooks/confhead/sitecustomize.py", T_HEADS),
    Lever("bigln_guard", "offload", FORWARD, EXACT, ("OF3O_LNSAFE",),
           "LayerNorm over ≥ 2^31 elements in row blocks (the CUDA kernel's 32-bit index guard), scoped to the model's own LayerNorm instances",
           "[of3o]", "offload/of3o/of3_offload.py:1450-1508", T_MODEL),
    Lever("host_pool", "offload", FORWARD, EXACT, ("OF3O_PIN_BUDGET_GB", "OF3O_PIN_POLICY"),
           "the pinned host buffer pool the row levers stream through (one buffer per tag; OF3O_PIN_BUDGET_GB the pinned total; a buffer over it is pageable with a NOTE line and a count under OF3O_PIN_POLICY=census, the lines' word, refused by name under `strict`; a pinned allocation the host refuses raises by name)",
           "[of3o]", "offload/of3o/of3_offload.py:81-132", T_MODEL),
    Lever("alloc_expandable", "offload", FORWARD, EXACT, ("PYTORCH_CUDA_ALLOC_CONF",),
           "the expandable-segments CUDA allocator for the kit process (fragmentation headroom at large N; the effective setting is recorded in the census)",
           "", "offload/README.md; offload/of3o/of3_offload.py:1555-1612", T_MODEL),
]}

# The canonical strategy id (opt_core/STRATEGIES.json `canonical[].id`; opt_core.report.lever_line validates it, an alias is refused) of the
# levers whose LEVER line names one: the confidence-phase levers of the big lines. report.lever_lines reads it; the tensor-parallel lever's id
# comes from opt_core.mem.ngpu there.
STRATEGY: Dict[str, str] = {
    "confhead": "F7.chunked_eval",          # the head evaluated per scorer row block: chunked evaluation of the confidence phase
    "conf_chunked": "F7.chunked_eval",      # the scorer over row blocks
}


# ---- GPU-class support (opt_core.arch: the one registry every lever consults; card_table() renders it) ----
ARCH_PREFIX = "LOCAL.openfold3_ob0."                 # kit-local lever ids in the arch registry (shared strategies are declared by their family module)
TESTED_SM = ("sm90",)                         # the GPU class every lever of this kit is tested on (H100)
MIN_SM = "sm80"                                  # the carried add-ons state sm80+ (stack.py MIN_CC; bf16/TF32 tensor-core paths)
# levers also tested on sm80 (A100, compute capability 8.0 — the SXM4 80GB and 40GB parts, configs/a100.env, the same pinned stack): the levers of the
# exact (cueq), fast, big·resident and big·tp (--n_gpu 2) lines, which is every lever of this kit — `exact --det 1` bitwise == `off --det 1`, the fast
# mode's pair cells served from the cc 8.0 rows of the core tables, big complete on one card and row-sharded on two. A lever added to a line joins this
# tuple once it has run on the card; until then card_table() names it outside the card's tested set.
A100_TESTED: tuple = ("ln_provider", "exactln", "fast_init", "cuda_graphs", "graphs_strict", "templ_distinct", "paircache", "trimul_v4", "triatt_block",
                      "pair_transition", "rollout_bf16", "dit_attn", "dit_glue", "token_agg", "atom_window", "atom_hoist", "apb_trunk", "templ_embed", "castcache",
                      "apb_hoist", "trunk_graph", "post_release", "tuner_guard", "conf_dtype", "z_dtype", "tp_shard_s", "tp_triatt", "tp_trimul", "sample_loop",
                      "trimul_hostsnap", "triatt_lean", "trans_inplace", "cond_once", "input_rows", "recycle_rows", "templ_host", "conf_chunked", "confhead", "bigln_guard",
                      "host_pool", "alloc_expandable", "triatt_exact", "trimul_exact", "transition_exact", "triatt_provider",
                      "postfwd_mem", "loader_workers", "fastjson")             # the runner-side cells, run on A100-SXM4-80GB under the composed one-GPU `big` line


def tested_sm(name: str) -> tuple:
    """The GPU classes lever `name` has test records on, ascending: sm80 (A100) for the A100_TESTED levers, then TESTED_SM (sm90) for all."""
    return (("sm80",) if name in A100_TESTED else ()) + TESTED_SM
# levers whose implementation is an image-bound compiled op: the class set is the image's (named, not excluded — another image of the same stack runs them)
ARCH_NOTES = {
    "trimul_v4": "triton_kernel_jit_compiled_per_sm;cells_keyed_cc|triton(opt_core/kernels/fpf_trimul_v4/table.json)",
    "triatt_block": "triton_kernels_jit_compiled_per_sm;core_cell_table(opt_core.attn.pair_fused_cells.json)",
    "pair_transition": "triton_kernel_jit_compiled_per_sm;core_cell_table(opt_core.attn.pair_fused_cells.json)",
    "rollout_bf16": "torch_autocast_bf16;no_kernel_of_its_own",
    "dit_attn": "triton_kernel_jit_compiled_per_sm(apb_attn|dtk_kernels.flash_bias_attn)",
    "dit_glue": "triton_kernels_jit_compiled_per_sm(dtk_kernels.ln_modulate|swiglu|gate_residual|apb_attn|flash_bias_attn)",
    "apb_trunk": "triton_kernel_jit_compiled_per_sm(apb_attn)",
    "templ_embed": "triton_kernels_jit_compiled_per_sm(templ_embed)",
    "token_agg": "triton_kernel_jit_compiled_per_sm(dtk_kernels.seg_reduce)",
    "atom_window": "triton_kernels_jit_compiled_per_sm(atom_window.ln_qkvg|window_attn)",
    "atom_hoist": "engine_statements_memoised;no_kernel_of_its_own",
    "castcache": "engine_statements_memoised;no_kernel_of_its_own",
    "exactln": "nvrtc_kernel_jit_compiled_per_sm(opt_core.kernels.ln exactln: cubin cached under MODEL_OPT_JIT_ROOT/exactln when writable, else compiled per process at install; variant triton jit_compiled_per_sm);core_cell_table(opt_core.kernels.ln LN_CELLS.json cc 9.0|8.0)",
    "ln_provider": "nvrtc_kernel_jit_compiled_per_sm(opt_core.kernels.ln exactln, as exactln) + triton_kernels_jit_compiled_per_sm(opt_core.kernels.ln rows fastln / ln_rows: Triton's own cache under TRITON_CACHE_DIR, compiled per process otherwise; each signature's first call off any capture)",
    "triatt_provider": "core_cell_table(opt_core.kernels.triattn.TRIATTN_CELLS.json:cc9.0+8.0);triton_rows_jit_compiled_per_sm",
    "trimul_provider": "core_cell_table(opt_core.kernels.trimul.TRIMUL_CELLS.json:cc9.0+8.0);triton_rows_jit_compiled_per_sm(tmk3,esm_v5_fwd:tl.make_tensor_descriptor>=triton3.6);kit_row_v4_per_cc_cells",
    "triatt_exact": "core_cell_table(opt_core.kernels.triattn.TRIATTN_CELLS.json);stock_library_call_by_name;exact_row_only_if_bit_proven_per_class",
    "trimul_exact": "core_cell_table(opt_core.kernels.trimul.TRIMUL_CELLS.json:cc9.0+8.0);tmk3_exact_triton+cublas(bit_proven_per_class);line_by_name_elsewhere",
    "trimul_form": "core_exact_tier_form_of3_module->row_of3_form(opt_core.kernels.trimul.of3_form:cublas_gemms+aten_or_exactln_layernorm,no_arch_specific_kernel;form_cells_cc9.0);bit_proven_per_class_and_shape_in_process;module_by_name_on_a_differing_shape",
    "transition_exact": "core_transition_provider(opt_core.kernels.transition:tier_word_exact,form_swiglu;cells+floors=TRANSITION_CELLS.json:cc9.0+8.0,per_stack_vouch);triton_kernel_jit_compiled_per_sm;stock_winner_or_unvouched_cell_runs_the_module_by_name;bitwise_checked_sm90_sm80",
    "apb_hoist": "engine_statements_memoised;no_kernel_of_its_own",
    "trunk_graph": "cuda_graph_capture_of_the_engine_stack;no_kernel_of_its_own",
    "post_release": "allocator_and_graph_state_release;no_kernel_of_its_own",
    "tuner_guard": "engine_tuner_comparison_guarded;no_kernel_of_its_own",
    "fastjson": "host_side_json_text_rendering;no_kernel_of_its_own;no_gpu_code",
    "writer_overlap": "host_side_output_writing_overlapped_with_the_next_item;no_kernel_of_its_own;no_gpu_code",
    "hostfeat": "host_side_featurisation_in_numpy;no_kernel_of_its_own;no_gpu_code",
    "ckpt_mmap": "host_side_checkpoint_read;no_kernel_of_its_own;no_gpu_code",
    "postfwd_mem": "engine_scoring_statements_blockwise;no_kernel_of_its_own",
    "loader_workers": "engine_dataloader_worker_count;no_kernel_of_its_own",
    "conf_dtype": "torch_autocast_bf16;no_kernel_of_its_own",
    "tp_shard_s": "collectives_plus_the_engine_statements;chunk_plan_is_the_80GB_card_table(tp.CHUNK_PLAN)",
    "sample_loop": "engine_statements_on_fewer_samples;no_kernel_of_its_own",
}


# the exact/ds4sci line's stack requirement (a runner-yaml route, not a registry lever): DeepSpeed's evoformer attention op is compiled per image


def declare_arch() -> dict:
    """Declare every lever of this kit in opt_core.arch (idempotent: the same content twice is a no-op): tested on tested_sm(name) (sm90 for
    all, + sm80 for A100_TESTED), floor MIN_SM, no exclusions known; the note names an image/JIT dependence where there is one. Returns {id: Support}."""
    from opt_core import arch
    out = {}
    for name in LEVERS:
        out[name] = arch.declare(ARCH_PREFIX + name, certified=tested_sm(name), min_sm=MIN_SM, note=ARCH_NOTES.get(name))
    return out


def card_table(sms=None) -> dict:
    """``{lever: {sm: word}}`` for this kit's levers over the arch registry's classes (CHANGES.md 'Cards')."""
    from opt_core import arch
    declare_arch()
    return {name[len(ARCH_PREFIX):]: row for name, row in arch.card_table([ARCH_PREFIX + n for n in LEVERS], sms).items()}
