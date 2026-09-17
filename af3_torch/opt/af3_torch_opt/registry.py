"""The one lever registry: what each lever is, its numerics class, its switch and the kit files it lives in.

Descriptive only — the package applies no lever itself: a lever is a name the kit's own ``build_model(levers=...)`` accepts
(``opt/forward/af3t/af3_torch/af3_torch_api.py``, the ``LEVER_SETS`` modes.py reads) or the DTK swap the carried adapter performs
(``forward.py`` ``XfoldDTK``, the in-model adapter over ``opt/forward/dtk``). Every path is relative to ``opt/forward/``;
a ``core:`` path is relative to the shared core's package (``common/opt_core/opt_core/``) — the module a ROUTED kernel executes from
(``KERNEL_ROUTES``). ``family`` is the cross-engine strategy family the lever belongs to (F1 triangle attention · F2 triangle multiplication ·
F3 capture · F4 precision · F5 attention backends · LOCAL engine-specific).
"""
from __future__ import annotations

KIT = "af3t"      # opt/forward/af3t: the model, api and kernel tree
PKG = "../af3_torch_opt"   # the package's own modules, relative to opt/forward/ like every `touches` entry (stack.touch_path)
DTK = "dtk"       # opt/forward/dtk:  the DTK fused diffusion transformer
CORE = "core:"    # prefix of a path inside the shared core's package (common/opt_core/opt_core/)

LEVERS = {
    "bf16w": {"name": "weights pre-cast to bf16", "kind": "runtime", "family": "F4", "numerics": "bitwise vs the eager port (`--mode off --nofastnn`: autocast casts each fp32 weight to the same bf16 bits per call); NOT bitwise vs the stock kernels (xfold's fastnn gated-linear-unit kernel reads the fp32 weight itself) — H100, BITWISE_EVIDENCE",
              "switch": "build_model(levers=…) 'bf16w'", "touches": [f"{KIT}/af3_torch/af3_torch_api.py"]},
    "trimul": {"name": "fused triangle multiplication served by the shared core's TriMul provider (opt_core.kernels.trimul) at c=128 by the MODE's tier word (fast in fast, big in big): the provider's measured cell per (card, precision, width, size bucket, direction) names the row on the running stack and its face serves it with the module's ten tensors; a row that refuses with the tensors in hand steps aside inside the face to the cell's next measured row (counted stepaside:<from>-><to>); a cell naming a library / torch statement, or a refusal nothing serves, = the module's own statement by name (fallback stock_row:<row> / refused); no row word and no launch cell of the kit's", "kind": "kernel", "family": "F2",
               "numerics": "tier 2 (the kernels line: fp64 error ratios at or below stock's per op vs the eager path, not byte-equal to stock, byte-equal run-to-run — the kit's own statement, af3t/kernels/af3_kernels.py:12-13)",
               "switch": "af3_kernels.enable(['trimul'])", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{CORE}kernels/trimul/"]},
    "triattn": {"name": "triangle attention on the shared core's triangle-attention provider (opt_core.kernels.triattn): the starting / ending attention of every pair stack -- trunk, MSA module, confidence head (head dim 32) and the template embedder (head dim 16) -- asks the provider for the MODE's tier word (fast in fast, big in big) at each (card, dtype, head dim, heads, size) cell and the provider serves its measured kernel row for the card; the kit names no row, tile or size table; a shape no kernel row serves on the card keeps the stock statement, by name (census refused:<row>:<reason>)", "kind": "kernel", "family": "F1", "numerics": "tier 2 (same class; the provider's tolerance-class rows: not bitwise to one another or to stock, bitwise run-to-run)",
                "switch": "af3_kernels.enable(['triattn'])", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{CORE}kernels/triattn/", f"{CORE}kernels/flash_triattn.py", f"{CORE}kernels/fpf_triatt_k2b/"]},
    "resid_fold": {"name": "Pairformer-block residual adds folded into the kernel epilogues: `pair += TriangleMultiplication(pair)` (x2) and `pair += Transition(pair)` run as one kernel each (fpf_trimul_v4 K3 / the transition provider row's epilogue, residual=True); off = the separate in-place adds", "kind": "kernel", "family": "F5",
                   "numerics": "bitwise the separate adds it replaces on the bf16 pair stream (torch.equal on the kit's adapters at 256-1216 tokens, H100); inside trimul / transition's bf16 fused-kernel class against off; an fp32 pair stream keeps torch's add (census residual_unfused)",
                   "switch": "af3_kernels.enable([..., 'resid_fold']) beside trimul / transition", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{CORE}kernels/fpf_trimul_v4/", f"{CORE}kernels/transition/"]},
    "attn_epi": {"name": "fused triangle-attention epilogue: the block's `pair += TriangleAttention(pair)` runs the attention's gate (sigmoid(g) * o), output projection and the residual add as one kernel (the shared core's fpf_triatt_epi triatt_epilogue, routed) with the launch cell the core's pair-fused cell table serves for the card (opt_core.attn.pair_fused lookup_cell); a card its table serves no tuned cell for, or an fp32 pair stream = the separate gate / projection / add (named)", "kind": "kernel", "family": "F5",
                 "numerics": "bitwise the statements it replaces (torch.equal vs lnl_fused.gate_transpose + F.linear + the in-place add on the kit's adapter at 256-1216 tokens, both directions, H100; the multimer identity input's five samples bitwise whole-forward against attn_epi off); inside triattn's bf16 class against off",
                 "switch": "af3_kernels.enable([..., 'attn_epi']) beside triattn", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{CORE}kernels/fpf_triatt_epi/", f"{CORE}attn/pair_fused_cells.json"]},
    "tmpl_trimul": {"name": "template pair-stack TriMul through the provider: the template embedder's 64-channel Pairformer blocks (2 blocks x every template x every trunk pass) ask the shared core's TriMul provider for the MODE's tier word at (c_z 64, c_hidden 64) beside lever trimul — the provider's measured 64-channel cell per card / size / direction names the row (fused rows on both cards), served through its face; off = the 64-channel module statement (census fallback:c=64 on lever trimul, as lever trimul alone)", "kind": "kernel", "family": "F2",
            "numerics": "tier 2 (lever trimul's class at 64 channels: the provider's rows are admitted per cell inside the stock bf16-autocast statement's error class vs an fp64 reference of the module; bitwise run-to-run; not bitwise to off)",
            "switch": "af3_kernels.enable([..., 'tmpl_trimul']) beside trimul", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{CORE}kernels/trimul/"]},
    "trimul_exact": {"name": "exact-class triangle multiplication through the provider: the pair stacks' TriangleMultiplication (c 128 trunk / MSA-module / confidence, c 64 template) asks the shared core's TriMul provider for its `exact` word in this engine's module form (form af3t_module -> row af3t_form: the module statement issued whole-tensor in a measured layout) and is served ONLY where the provider's table vouches that row byte-identical to the module for the running stack, width and size (today: H100 cc 9.0 on torch 2.13 / triton 3.7, bf16 and fp32-in-autocast pair streams, c 128 and 64, every size up to 1200 tokens); any other stack or size (an A100; items above 1200 tokens) is refused BY NAME per shape and the module's own statement runs (package lever tri_layout's restatement where that lever is on), census fallback:unvouched; under fast / big lever trimul's tier word owns the class and this lever steps aside per call by name (fallback:superseded:trimul)", "kind": "kernel", "family": "F2",
                     "numerics": "bitwise vs off where served (the provider's per-stack vouch records: the row equals the module byte for byte at every admitted size; outside them the module itself runs) — H100; on the A100 every call is the module's (refused by name)",
                     "switch": "af3_kernels.enable(['trimul_exact'])", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{CORE}kernels/trimul/", f"{CORE}kernels/trimul/af3t_form.py"]},
    "pwa_lnl": {"name": "MSA pair-weighted averaging: the MSA module's MSAAttention takes `pair_norm(pair) -> pair_logits -> permute` ([N,N,128] LayerNorm, 8-logit projection) from one fused LayerNorm + projection kernel (the kit's lnl_fused.ln_linear, head-plane-major output: the [8,N,N] logits contiguous); every other statement of the module verbatim", "kind": "kernel", "family": "F5",
            "numerics": "tier 2 (the two stock statements' own arithmetic class under bf16 autocast — fp32 LayerNorm, bf16 products accumulated in fp32, one bf16 rounding per logit — in another reduction order: 0.015 % of logits differ by one bf16 ulp, module output max abs 5e-4, error vs an fp64 reference 1.000x the stock statements' (448-1216 tokens, 1024 rows, H100); bitwise run-to-run; not bitwise to off)",
            "switch": "af3_kernels.enable([..., 'pwa_lnl'])", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{KIT}/kernels/third_party/lnl_fused.py"]},
    "pwa_msa": {"name": "the MSA module's pair-weighted averaging, msa side: `act_norm(msa)` -> `v_projection` and `-> gating_query` (LayerNorm over [S, N, 64] msa rows, two 64->64 projections) as ONE LayerNorm+projection kernel (kernels/third_party/lnl_fused.py ln_linear, 64 -> 128 columns: values 0..63, gate logits 64..127; LN in fp32, the normalised row rounded to bf16, bf16 x bf16 products accumulated in fp32, one bf16 store) -- the same arithmetic class as the stock statements under bf16 autocast, not the same bits; per call 0.067 / 0.118 / 0.169 ms vs 0.985 / 1.823 / 2.645 ms at 448 / 832 / 1216 tokens on an H100 (44 calls per item); every other statement verbatim; fast / big only", "kind": "kernel", "family": "F5",
            "numerics": "tier 2 (the two stock statements' own arithmetic class under bf16 autocast — fp32 LayerNorm, bf16 products accumulated in fp32, one bf16 rounding per logit — in another reduction order: 0.015 % of logits differ by one bf16 ulp, module output max abs 5e-4, error vs an fp64 reference 1.000x the stock statements' (448-1216 tokens, 1024 rows, H100); bitwise run-to-run; not bitwise to off)",
            "switch": "af3_kernels.enable([..., 'pwa_lnl'])", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{KIT}/kernels/third_party/lnl_fused.py"]},
    "opm": {"name": "MSA outer-product mean: the MSA module's OuterProductMean on the kit's af3t_opm kernels — LayerNorm + left|right projections + mask in one pass over the MSA stream, the outer product one bf16 GEMM per chunk of 256 left tokens written straight into the layout the output contraction reads (the stock statement's [N,N,32x32] intermediate is never permuted or copied and exists one chunk at a time), the 1024-long output contraction + bias + mask-norm division in one kernel with the Evoformer block's `pair += outer_product_mean(msa)` folded into its epilogue (pair rows updated in place; bitwise `pair += update` on the kernel's own update); the mask norm computed once per mask tensor", "kind": "kernel", "family": "F5",
            "numerics": "tier 2 (the stock statements' own arithmetic class under bf16 autocast — fp32 LayerNorm, bf16 GEMM operands and outputs, fp32 accumulation, fp32 epilogue with the statement's bf16 rounding of eps + norm — in another reduction order: module output max abs 2.6e-3 from the stock statements with 4.4-4.8 % of elements differing at all, error vs an fp64 reference 1.000x the stock statements' in max and rms (448-1216 tokens, 1024 rows, H100); bitwise run-to-run; not bitwise to off)",
            "switch": "af3t_msa.enable(['opm'])", "touches": [f"{KIT}/kernels/af3t_msa.py", f"{KIT}/kernels/third_party/af3t_opm.py"]},
    "transition": {"name": "fused LayerNorm + SwiGLU + Linear transition served by the shared core's transition provider (opt_core.kernels.transition) asked by the MODE's tier word (fast in fast, big in big): per (cc, dtype, c, hidden, tokens) cell the provider serves the row it measured fastest in class on the card (v2 / v1 / af3_fused / lnl ...: one carried copy each in the core, launched with the cell's measured launch word; no candidate tile timed, no copy or launch table in this tree), names the engine's own module where that measured fastest (counted fallback:c=<c>,stock-row) or refuses by name (no cell for the shape on the card: fallback:c=<c>,<kind>); shapes: pair c=128x4, MSA c=64x4, template c=64x2, single c=384x4", "kind": "kernel", "family": "LOCAL", "numerics": "tier 2 (same class: bf16 operands, fp32 accumulation, the stock graph's rounding points; accumulation order differs from cuBLAS's); never in exact (exact's transition = xfold's own kernels + lever glu_proj, bitwise)",
                   "switch": "af3_kernels.enable(['transition'])", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{CORE}kernels/transition/"]},
    "glu_proj": {"name": "the transition's gated linear unit + output projection (+ the block's residual add) as one Triton kernel, byte-equal to xfold's fastnn statement (checked per shape in process)", "kind": "kernel", "family": "LOCAL",
                 "numerics": "bitwise (the stock kernels' rounding points and accumulation order; a shape whose cuBLAS statement differs is served by stock, by name)",
                 "switch": "af3_kernels.enable(['glu_proj'])", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{KIT}/kernels/af3t_glu_proj.py"]},
    "apb": {"name": "attention with pair bias through the shared core's pair-bias attention provider by the mode's tier word (fast | big: its measured row per card / cell / size -- its Triton kernel, a carried package or SDPA -- instead of eager softmax; the Pairformer's pair logits through its LayerNorm + projection producer rows)", "kind": "kernel", "family": "F5",
            "numerics": "tier 2 (the SDPA class: bf16 operands, fp32 softmax / accumulation; a call class the provider's row refuses by name is served by the kit's SDPA statement, named)", "switch": "af3_kernels.enable(['apb'])", "touches": [f"{KIT}/kernels/af3_kernels.py", f"{DTK}/dtk_modules.py", f"{CORE}kernels/apb/__init__.py", f"{CORE}kernels/apb_attn.py"]},
    "stepgraph": {"name": "diffusion-head hoist of step-invariant conditioning + whole-denoiser-step CUDA graph", "kind": "runtime", "family": "F3",
                  "numerics": "same arithmetic, computed once per sample / replayed per step (byte-equal to stock — the stock kernels, unpadded — and to the eager port; H100)", "switch": "build_model(levers=…) 'stepgraph'",
                  "touches": [f"{KIT}/af3_torch/xfold/nn/diffusion_head.py", f"{KIT}/af3_torch/af3_torch_api.py"]},
    "hoist": {"name": "diffusion-head hoist of the step-invariant conditioning (pair conditioning, DiT pair logits, atom encoder/decoder statics) without the whole-step graph — what big's graph_drop keeps of stepgraph", "kind": "runtime", "family": "F3",
              "numerics": "same arithmetic as stepgraph's hoist, computed once per sample (byte-equal to stock and to the eager port; H100)", "switch": "build_model(levers=[..., 'hoist'])", "touches": [f"{KIT}/af3_torch/af3_torch_api.py", f"{KIT}/af3_torch/xfold/nn/diffusion_head.py"]},
    "compile": {"name": "torch.compile (max-autotune-no-cudagraphs) of the atom-transformer / transition / conditioning glue inside the step graph",
                "kind": "runtime", "family": "F3", "numerics": "tier 2 (same bf16 class, not bitwise vs the uncompiled set; the first call at a token length compiles for a minute or two)",
                "switch": "build_model(levers=…) 'compile' (enable_compile)", "touches": [f"{KIT}/af3_torch/af3_torch_api.py"]},
    "sbatch": {"name": "sample-batched diffusion sampler: every diffusion sample advanced together per step (T batched denoiser calls instead of S x T serial ones), the hoisted statics and the step's single conditioning shared by the samples, the batched step one CUDA graph under stepgraph, stock's random draws draw for draw",
               "kind": "runtime", "family": "LOCAL",
               "numerics": "bitwise vs the serial sampler of the same lever set when the step's kernels are the same at batch S as at batch 1 — measured so with `compile` off on both routes (DTK: the periodic-row kernels + ONE launch of the shared core's pair-bias kernel apb_attn for all samples == the per-sample flash kernel's bits; xfold: the class's own statements over the sample axis) at 448 tokens on H100: every GEMM / row kernel / attention program computes a sample's rows from that sample's rows only; under `compile` the tolerance class (the compiled glue specialises its kernels by rank: reduction order); the augmentation, the noise and the update are stock's per-sample statements on stock's own draws (RNG draws = stock's: diffusion_head.DrawPlan seeks the Philox stream per (sample, step) — 3001/3001 draws of a 5-sample trajectory equal bit for bit)",
               "switch": "build_model(levers=…) 'sbatch' (AlphaFold3.sample_batch = b: at most b samples per denoiser call; 0 = the serial sampler)",
               "touches": [f"{KIT}/af3_torch/xfold/alphafold3.py", f"{KIT}/af3_torch/xfold/nn/diffusion_head.py", f"{KIT}/af3_torch/xfold/nn/diffusion_transformer.py",
                           f"{KIT}/af3_torch/xfold/nn/atom_cross_attention.py", f"{DTK}/dtk_modules.py", f"{KIT}/af3_torch/af3_torch_api.py", f"{PKG}/canonical_noise.py",
                           CORE + "attn/apb_core.py", CORE + "kernels/apb_attn.py"]},
    "atom_window": {"name": "the atom transformers' sequence-local attention (atom-attention encoder / decoder: 3 + 3 blocks x every denoiser step over the samples) on the support "
                            "library's two fused window kernels (opt_core.kernels.atom_window: LayerNorm + both AdaLN modulations from per-atom conditioning rows + the q|k|v|gate "
                            "projections in one pass with k / v once per atom instead of on gathered 128-key window copies; then, per 32-query block and sample, the shifted key window "
                            "read in place, pair bias + validity mask, fp32 softmax, p v, sigmoid gate, W_o, AdaLN-Zero gate and residual in one kernel); the step-invariant operands "
                            "hoisted per trajectory; only the query blocks that hold real atoms are computed",
                    "kind": "kernel", "family": "F5",
                    "numerics": "the kernels' stated class: fp32 in / out, TF32 round-to-nearest tensor-core dots (cuBLAS-TF32 class), fp32 softmax — at or inside the bf16-autocast route's "
                                "distance from an fp32 reference (tests/test_atom_window.py); not bitwise vs off",
                    "switch": "build_model(levers=…) 'atom_window' (enable_atom_window: DiffusionHead.use_atom_window + DiffusionCrossAttTransformer.WINDOW_KERNEL); steps aside by name without the hoist",
                    "touches": [f"{CORE}kernels/atom_window.py", f"{KIT}/af3_torch/xfold/nn/diffusion_transformer.py", f"{KIT}/af3_torch/xfold/nn/diffusion_head.py",
                                f"{KIT}/af3_torch/xfold/nn/atom_cross_attention.py", f"{KIT}/af3_torch/af3_torch_api.py"]},
    "token_agg": {"name": "the atom-attention encoder's atom -> token aggregation (gather of the projected atom activations to the token-atoms layout, gather mask, relu, masked mean "
                          "over each token's atoms: five passes over a [samples, tokens, 24, 768] intermediate per denoiser step) as ONE kernel that reads each token's atom rows "
                          "once and writes the [samples, tokens, 768] mean (af3t_token_agg, carried beside the DTK modules); its step-invariant operands (slot weights, "
                          "1 / atoms per token, gather rows) hoisted per trajectory with the encoder statics",
                  "kind": "kernel", "family": "F5",
                  "numerics": "fp32 accumulation over a token's 24 atom slots in slot order (torch's sum has its own order): |diff| ~ 1e-6 relative to the stock statements "
                              "(tests/test_token_agg.py); sample s of a batched call is bitwise the single-sample call; not bitwise vs off",
                  "switch": "build_model(levers=…) 'token_agg' (enable_token_agg: AtomCrossAttEncoder.TOKEN_AGG); refuses a call by name (census) -> the stock statements",
                  "touches": [f"{DTK}/af3t_token_agg.py", f"{KIT}/af3_torch/xfold/nn/atom_cross_attention.py", f"{KIT}/af3_torch/af3_torch_api.py"]},
    "atom_rows": {"name": "the atom transformers' three blocks — window kernels AND transition — on ONE contiguous fp32 working copy of the real query blocks' rows "
                          "(32 * ceil(real atoms / 32) of the padded layout's rows: about a third at the ladder's sizes), cast back into the activation once; the "
                          "padded rows keep their input values (both callers zero them by the atom mask right after); no per-block contiguity copies, the transition's "
                          "GEMMs / AdaLN / SwiGLU / residual on the real rows only",
                  "kind": "runtime", "family": "LOCAL",
                  "numerics": "the same row-local statements on fewer rows: the compiled transition (dynamic row count) may pick other GEMM algorithms than on the padded "
                              "layout — the tolerance class (tests/test_atom_window.py GPU: rows path vs full path inside the bf16 route's class); not bitwise vs off",
                  "switch": "build_model(levers=…) 'atom_rows' (enable_atom_rows: DiffusionCrossAttTransformer.WINDOW_ROWS_ONLY; needs atom_window -> 'needs_atom_window' by name); a call whose layout it cannot take is named (census) and runs forward_windowed",
                  "touches": [f"{KIT}/af3_torch/xfold/nn/diffusion_transformer.py", f"{KIT}/af3_torch/af3_torch_api.py"]},
    "prologue": {"name": "the sample-batched denoiser step's prologue (per sample: random rigid augmentation of the positions + the step's noise) with every random draw made "
                         "as the serial statements make it (same calls, same Philox positions, same bits, generator left where they leave it) and the arithmetic — Gram-Schmidt, "
                         "centring, the fp32 rigid transform, mask, noise lay-up and add — restated once over the sample axis (~12 launches per step instead of ~60)",
                 "kind": "runtime", "family": "LOCAL",
                 "numerics": "noise term bit-identical (same draws); the batched norm / dot / einsum of the augmentation may differ from the per-sample statements in the last "
                             "bit — the tolerance class; RNG stream identity kept draw for draw (diffusion_head.DrawPlan)",
                 "switch": "build_model(levers=…) 'prologue' (AlphaFold3.batched_prologue; needs sbatch -> 'needs_sbatch' by name)",
                 "touches": [f"{KIT}/af3_torch/xfold/alphafold3.py", f"{KIT}/af3_torch/xfold/nn/diffusion_head.py", f"{KIT}/af3_torch/af3_torch_api.py"]},
    "dtk": {"name": "DTK FusedDiT: the 24-block diffusion token transformer as cuBLAS GEMMs with concatenated weights + fused Triton row kernels + flash attention with hoisted bf16 pair bias",
            "kind": "kernel", "family": "LOCAL", "numerics": "tier 2 (module level: error ratio DTK/stock vs fp64 below 1 for the token and the atom transformer, bitwise run-to-run; the in-model bracket on the fastest set, §3)",
            "switch": "--dtk 1 (forward.py XfoldDTK swap)", "touches": [f"{DTK}/dtk_modules.py", f"{PKG}/forward.py", f"{CORE}kernels/dtk_kernels.py"]},
    "template_dedupe": {"name": "template distinct-slot evaluation: the template embedder runs its pair stack once per DISTINCT template slot and adds the result once per slot in stock order (an untemplated input carries 4 byte-identical dummy slots)",
                        "kind": "runtime", "family": "LOCAL", "numerics": "byte-equal to stock by construction (the same addends in the same order; H100)",
                        "switch": "forward.py --package-levers template_dedupe (template_dedupe.install; the package's, composed on every mode but off)", "touches": [f"{PKG}/template_dedupe.py", f"{KIT}/af3_torch/xfold/nn/template.py"]},
    "dev_scalars": {"name": "device-resident scalars: the trunk pass's constant scalars — the vector norm's epsilon clip (xfold geometry Vec3Array.norm, three calls per template slot evaluation) and the bond contact matrix's ones / zero (Evoformer._embed_bonds) — live on the device instead of being copied from the host at every use (each copy a transfer the stream synchronises on)",
                    "kind": "runtime", "family": "F6", "numerics": "byte-equal to stock by construction (the same operand values: the epsilon tensor is the stock statement's, built once; 1.0 / 0.0 written at the same indices; H100)",
                    "switch": "forward.py --package-levers dev_scalars (dev_scalars.install; the package's, composed on every mode but off)", "touches": [f"{PKG}/dev_scalars.py", f"{KIT}/af3_torch/xfold/geometry.py", f"{KIT}/af3_torch/xfold/alphafold3.py"]},
    "tri_layout": {"name": "triangle-multiplication operand layout: the stock triangle multiplication's three big layout copies — the two GEMM operands torch's einsum clones out of the interleaved channel-major view, and the product's transpose into the layer norm — are written by one tiled transpose kernel each (coalesced on both sides) instead of torch's generic strided copy; the GEMM and the layer norm receive the same bytes in the same layouts (the trunk's largest kernels at 800 tokens on the stock path)",
                   "kind": "runtime", "family": "F2", "numerics": "byte-equal to stock by construction (data movement only: the same operand bytes in the layouts stock's clones produce, so the same cuBLAS call and the same layer-norm kernel run on them; H100)",
                   "switch": "forward.py --package-levers tri_layout (tri_layout.install; the package's, composed on every mode but off: the class forward where it is still the stock one — exact — else the trimul kernel lever's by-name fallback to the stock forward, the template pair stack's c=64 rows under fast / big)", "touches": [f"{PKG}/tri_layout.py", f"{KIT}/af3_torch/xfold/nn/triangle_multiplication.py"]},
    "ln_rows": {"name": "row-blocked LayerNorm launch: xfold's fastnn Triton LayerNorm kernel — one program per row, 640 000 one-warp programs per pair-track call at 800 tokens — runs the same kernel body over 8 consecutive rows per program (the identical per-row code: bitwise by construction); pair, MSA, template and atom-pair rows up to 512 bytes wide are served (×2.1 per call on the pair track's bf16 rows, ×2.9 on the atom-pair rows), wider rows keep the stock launch by name (wide=)",
                "kind": "runtime", "family": "F5", "numerics": "byte-equal to stock by construction (the stock kernel's statements per row, in a row loop: the same layouts, reduction tree and conversions; torch.equal at every width / dtype / affine form / contiguity the model calls it with; H100)",
                "switch": "forward.py --package-levers ln_rows (ln_rows.install; the package's, composed on every mode but off: the fastnn LayerNorm class forward wherever a kernel lever has not absorbed the LayerNorm)", "touches": [f"{PKG}/ln_rows.py", f"{KIT}/af3_torch/xfold/fastnn/layer_norm.py"]},
    "attn_layout": {"name": "grid self-attention operand layout: the triangle attention's three head-major operands ([b, h, n, d], which the fastnn wrapper copies contiguous out of the projections' [b, n, h·d]) and its token-major output are written by a row-regroup kernel that moves whole 64-byte d-rows (loads and stores only) instead of torch's generic 2-byte-element strided copy (312 + 104 copies per trunk pass, 115 ms per pass at 800 tokens); the attention kernel receives the same bytes at the same strides; pair tracks under 512 tokens keep the stock copies by name (small=): there the pass is launch-bound and the regroup launches cost more than they save",
                    "kind": "runtime", "family": "F2", "numerics": "byte-equal to stock by construction (data movement only: the contiguous operands stock's .contiguous() makes and the contiguous output stock's rearrange makes; H100)",
                    "switch": "forward.py --package-levers attn_layout (attn_layout.install; the package's, composed on every mode but off: GridSelfAttention._attention where the class is still the kit's own — exact; under fast / big the triangle-attention kernel levers own the class and the lever is named aside)", "touches": [f"{PKG}/attn_layout.py", f"{KIT}/af3_torch/xfold/nn/attention.py", f"{KIT}/af3_torch/xfold/fastnn/attention.py"]},
    "gate_fuse": {"name": "fused gating glue: the pair stack's `x *= mask`, `torch.sigmoid(gate)`, `x *= sig` statements (triangle multiplication: two gates per call; grid self-attention: one) as one Triton kernel per group with the stock rounding points — the masked product rounded to bf16, the sigmoid on libdevice's expf (the function ATen's kernel calls) rounded to bf16, then the gate product — instead of three ATen passes over an [N, N, 128…256] tensor; composes on tri_layout / attn_layout, which call it where the stock statements stand",
                  "kind": "runtime", "family": "F2", "numerics": "byte-equal to stock (the stock statements' arithmetic and rounding points in one kernel; equal on every element of 6 140 live triangle-multiplication calls, masked and unmasked, bf16 and fp32 masks, 128 / 256 channels at 400 / 800 tokens; H100)",
                  "switch": "forward.py --package-levers gate_fuse (gate_fuse.install; the package's, composed on every mode but off where tri_layout / attn_layout serve: exact; under fast / big their hosts are aside or fallback-only and so is this)", "touches": [f"{PKG}/gate_fuse.py", f"{PKG}/tri_layout.py", f"{PKG}/attn_layout.py"]},
    "castcache": {"name": "autocast weight-cast memo: each fp32 nn.Linear keeps the bf16 weight / bias autocast would cast on every call, cast once per process (the same cast op, so the GEMM's operands are the same bits); the diffusion sampler's step graphs lose several hundred cast kernels per step — exact only: under fast / big the kit lever bf16w stores every Linear weight in bf16 and autocast casts nothing, so there is no cast to memoise there",
                  "kind": "runtime", "family": "F4", "numerics": "byte-equal to stock by construction (the memo holds the tensor autocast's own cast produces; the parameters stay fp32 in place, so the fastnn kernels that read them directly are untouched; H100)",
                  "switch": "forward.py --package-levers castcache (castcache.install; the package's, composed on exact only — see name)", "touches": [f"{PKG}/castcache.py"]},
    "canonical_noise": {"name": "canonical diffusion noise: the sampler's shape-dependent random draws (initial positions, per-step noise) are made at the input's own token count (the length stock runs at, unpadded) and laid into the model's padded length (padding rows zero, no draw), so the real atoms' noise is stock's for the seed whatever the kit pads to",
                        "kind": "runtime", "family": "LOCAL", "numerics": "the noise stock draws for the input, draw for draw (bitwise the stock statements' values when the model runs unpadded); the model itself runs at the kernel_tile length (fast's class)",
                        "switch": "forward.py --package-levers canonical_noise --padding kernel_tile (canonical_noise.install; n_gpu = 1)", "touches": [f"{PKG}/canonical_noise.py", f"{KIT}/af3_torch/xfold/alphafold3.py"]},
    "prefetch": {"name": "featurise-ahead: the process chain runs streamed — the featuriser loads every fold input, announces their seeds, and the model process is launched at once (the model built and the weights loaded while the first input featurises); input i+1 featurises on the CPU while input i is on the GPU, the model process taking each (input, seed) the moment its batch files are handed over (an empty marker written after they are closed). The same three processes, files and statements: scheduling only", "kind": "runtime", "family": "F6",
                 "numerics": "bitwise vs off by construction (the batch bytes the model process reads are the sequential chain's; nothing in the model changes) — H100, BITWISE_EVIDENCE",
                 "switch": "featurise.py --stream 1 + forward.py --stream-root <work> (cli.cmd_pred, pipeline_stream; n_gpu = 1, run_inference)", "touches": [f"{PKG}/pipeline_stream.py", f"{PKG}/cli.py", f"{PKG}/featurise.py", f"{PKG}/forward.py"]},
    "write_behind": {"name": "writer-behind: the fork's writers run beside the model process and write input i (mmCIF, confidences, ranking, terms-of-use stamps) while input i+1 is on the GPU, taking each input the moment every seed's result file is handed over (an empty marker written after it is closed); the last input's write is the only one left after the model process ends. The same writer functions and files: scheduling only", "kind": "runtime", "family": "F6",
                     "numerics": "bitwise vs off by construction (the result bytes the writers read are the sequential chain's) — H100, BITWISE_EVIDENCE",
                     "switch": "postprocess.py --follow <work> + forward.py --stream-root <work> (cli.cmd_pred, pipeline_stream; n_gpu = 1, run_inference)", "touches": [f"{PKG}/pipeline_stream.py", f"{PKG}/cli.py", f"{PKG}/postprocess.py", f"{PKG}/forward.py"]},
    "autotune_cache": {"name": "Triton autotune results kept on disk under the kit's Triton cache directory and reused by every later model process (Triton's own autotune cache, TRITON_CACHE_AUTOTUNING): the kit's autotuned kernels — xfold's gated-linear-unit kernel (off's kernels: exact), the fused transition and ln_linear kernels (fast, big) — are benchmarked over their tile configurations once per (kernel, key, arch, Triton version) instead of once per process, so the first item of a pred on a warm cache root skips the benchmarking (≈ 7–8 s per process at one input size on H100, again for every further size); a cold or cleared cache root re-measures and re-fills it", "kind": "runtime", "family": "F3",
                       "numerics": "bitwise vs off given the tuner's choice per key is stable — the assumption exact ≡ off across two processes already rests on (off re-benchmarks in every process); the cache pins the first measured choice — H100, BITWISE_EVIDENCE",
                       "switch": "TRITON_CACHE_AUTOTUNING=1 in the model process's environment (cli.cmd_pred) + triton.knobs.autotuning.cache before any kernel module is imported (forward.run); ablated by name like every package lever", "touches": [f"{PKG}/cli.py", f"{PKG}/forward.py"]},
    "feat_par": {"name": "parallel featurisers: under the streamed chain the featurise step runs as two processes of the fork's featuriser, the pred's inputs dealt between them round-robin, so two inputs featurise on the CPU at the same time while the GPU works — for inputs whose featurisation takes longer than their forward (a few hundred tokens under fast) the model process no longer waits on the one featuriser; each process announces its own inputs' seeds, the wrapper merges their reports into the one featurise report. A pred of one input, or one without prefetch, runs one featuriser (said on the LEVER line)", "kind": "runtime", "family": "F6",
                 "numerics": "bitwise vs off by construction (each input is featurised by the same statements with its own seeds, whichever process runs it) — H100, BITWISE_EVIDENCE",
                 "switch": "cli.cmd_pred: pipeline_stream.FEAT_WORKERS featurise.py --stream 1 processes over disjoint --item lists (requires prefetch; n_gpu = 1, run_inference)", "touches": [f"{PKG}/pipeline_stream.py", f"{PKG}/cli.py"]},
}

# The package levers: composed by this package inside the model process on top of the kit's build (forward.py --package-levers), on every
# mode but `off` (modes.MODE_PACKAGE_LEVERS). The kit's files are untouched; a package lever restates specific statements of the kit's own modules
# (tests/test_kit_statement_mirrors.py holds each restatement's source to the kit's, live, at test time).
PACKAGE_LEVERS = ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "castcache", "canonical_noise", "prefetch", "write_behind", "autotune_cache", "feat_par")

# EXACT: the kit levers `exact` keeps from `fastest` (modes.MODE_FILTER) — each byte-equal to the stock-kernels base (`off`: xfold's shipped
# Triton fastnn kernels, the stock CLI's --fastnn default, itself byte-equal across processes). `stepgraph` implies the hoist (build_model
# enables it under stepgraph). `bf16w` is bitwise against the eager port only: xfold's fastnn gated-linear-unit kernel reads the fp32 weight
# itself, so pre-cast weights change its arithmetic on the stock-kernels base — it stays a fast / big lever. BITWISE_EVIDENCE: each lever's
# byte-equality class against the eager `off` and against the stock-kernels base; `kept` = in force on a shipped line. NOT_BITWISE: the levers of
# the bf16 fused-kernel numerics class (not byte-equal to `off`).
EXACT = ("stepgraph", "glu_proj", "trimul_exact")
BITWISE_EVIDENCE = {
    "bf16w": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": False, "kept": True},
    "stepgraph": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "glu_proj": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "trimul_exact": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True, "why": "served only inside the provider's vouch records for the running stack (row af3t_form byte-identical to the module at every admitted size: H100, torch 2.13 / triton 3.7, c 128 / 64, N <= 1200); every other call is the module's own statement, refused by name"},
    "hoist": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": False, "why": "implied by stepgraph (build_model enables the hoist under stepgraph)"},
    "template_dedupe": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "dev_scalars": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "tri_layout": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "ln_rows": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "attn_layout": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "gate_fuse": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "castcache": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "canonical_noise": {"bitwise_vs_off": True, "kept": False, "why": "a no-op without padding (off / exact: the model length is the token count); it rides the kernel_tile policy of fast / big"},
    "prefetch": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "write_behind": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
    "autotune_cache": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True, "why": "the cached choice is the tuner's own first measurement per key; exact's equality to off already assumes that choice is stable per key across processes"},
    "feat_par": {"bitwise_vs_off": True, "bitwise_vs_stock_kernels": True, "kept": True},
}
NOT_BITWISE = ("trimul", "triattn", "transition", "apb", "resid_fold", "attn_epi", "tmpl_trimul", "pwa_lnl", "opm", "pwa_msa", "compile", "sbatch", "atom_window", "token_agg", "atom_rows", "prologue", "dtk")   # the bf16 fused-kernel class: not byte-equal to off (resid_fold / attn_epi are bitwise to the kernels they fold into, not to off)


# The kernels the model process ROUTES to the shared core's carried copies (opt_core.kernels.route; common/opt_core/opt_core/kernels/<name>).
# `kit_copy`: the kit's own carried copy of the module (shadowed by the route for exactly these names; byte-identical to the core's for every
# file the core's META record lists, or named in KERNEL_PARITY — tests/test_core_adoption.py holds the bytes equal), or None: the core's
# copy is the only one (flash_triattn, and dtk_kernels: the DTK add-on's own v0.1 module left the tree at 0.2.12, shadowed by this route in every mode since the route exists;
# its bytes are kept byte for byte as a reference copy outside the release tree, beside this engine's definitions: refs/dtk_kernels_v01_reference.py).
# Before the kit's api is imported, forward.py routes each name and gates it with opt_core.kernels.route_check (the resolved
# bytes == the sums file; every export present) — a refused route is exit 2 of the model process, never the kit copy served silently.
# `exports`: this kit's data the core copy does not carry, by the sums file's parameter name -> the kit file (relative to opt/forward/),
# exported into the model process's environment (stack.kernel_exports). `levers`: the levers whose calls execute the routed module.
# Not routed, by decision: the TriMul kernels (fpf_trimul_v4, the provider's native / ESM-lineage rows, af3t_form): levers trimul / tmpl_trimul / trimul_exact
# reach them only through the shared core's provider face (opt_core.kernels.trimul resolves its own carried modules); no kit code imports those names and
# the kit carries no cell table for them.
KERNEL_ROUTES = {
    "flash_triattn": {"levers": ("triattn",), "exports": {}, "kit_copy": None},          # the provider's Triton triangle-attention rows (flash, K2B) execute from these routed core modules where the
    "fpf_triatt_k2b": {"levers": ("triattn",), "exports": {}, "kit_copy": None},         #   provider's cell for the card names them (lever triattn; opt_core.kernels.triattn imports them through the same route)
    "fpf_triatt_epi": {"levers": ("attn_epi",), "exports": {}, "kit_copy": None},         # the fused gate * o -> out-projection -> residual epilogue (lever attn_epi): imported where the core's pair-fused table serves a cell for the card
    "dtk_kernels": {"levers": ("dtk",), "exports": {}, "kit_copy": None},          # the DTK row kernels dtk_modules.py imports (`import dtk_kernels`): the core's copy is the only one (0.2.12)
    "apb_attn": {"levers": ("sbatch", "apb"), "exports": {}, "kit_copy": None},   # the shared core's own pair-bias attention kernel: a row its provider serves lever apb and the sample-batched DTK step by the tier word, and the batched step's direct entry (opt_core.attn.apb_core)
    "atom_window": {"levers": ("atom_window",), "exports": {}, "kit_copy": None},  # the sequence-local atom-attention kernels (ln_qkvg + window_attn) of lever atom_window
}
# A routed kernel whose core copy is NOT byte-identical to the kit's carried copy: {name: {file: proof}} naming why the served (core) bytes
# still keep the lever's class word. Checked LIVE at test time (test_core_adoption.py: got != want, never a stored digest of either file).
# Every other routed file is byte-identical.
# (empty since 0.2.12: the one entry, dtk_kernels.py — "opt_core CHANGES 0.5.17.0 `dtk_kernels` periodic operands: every row kernel called with the
# new keyword arguments at their defaults (this kit passes none of them) is BITWISE the DTK v0.1 module — common/opt_core/tests/gpu/test_dtk_kbg1_gpu.py
# with DTK_REF_MODULE naming that module" — left with the kit copy; the v0.1 reference the GPU proof names now lives outside the release
# tree, beside this engine's definitions: refs/dtk_kernels_v01_reference.py, importable as DTK_REF_MODULE=dtk_kernels_v01_reference.)
KERNEL_PARITY: dict = {}


# The activation evidence of a lever in one pred's records (cli.py prints one LEVER line per registry lever per pred; report.lever_line):
# where in the model process's report the proof that the lever RAN is read — `applied`: the name is on the model's own `_af3t_levers`
# record (levers_applied); `census`: the kit's per-call census carries served/fallback counters under the lever's name; `graph_captures`:
# whole-step graphs captured (summed over items); `compiled`: the model's `_af3t_compiled` record; `dtk`: the swap record (dtk, dtk_swap_s).
EVIDENCE = {"bf16w": "applied", "trimul": "census", "trimul_exact": "census", "triattn": "census", "transition": "census", "glu_proj": "census", "apb": "census", "resid_fold": "census", "attn_epi": "census", "tmpl_trimul": "census", "pwa_lnl": "census", "pwa_msa": "census", "opm": "census",
            "stepgraph": "graph_captures", "hoist": "applied", "compile": "compiled", "sbatch": "applied", "atom_window": "census", "token_agg": "census", "atom_rows": "census", "prologue": "applied", "dtk": "dtk",
            "template_dedupe": "package", "dev_scalars": "package", "tri_layout": "package", "ln_rows": "package", "attn_layout": "package", "gate_fuse": "package", "castcache": "package", "canonical_noise": "package", "prefetch": "package", "write_behind": "package", "autotune_cache": "package", "feat_par": "package"}      # package: forward.json package_levers_applied + the lever's own census (template_dedupe: calls / slots / evaluated; dev_scalars: norm calls / eps tensors / bond calls; tri_layout: calls / tiled / generic / scope; ln_rows: calls / served / wide / generic; attn_layout: calls / regrouped / small / generic; gate_fuse: calls / fused / generic / form; castcache: linears / mib / served / stock; canonical_noise: model / canonical lengths)
# The pinned impl= / origin= slots of every LEVER line (opt_core.report.lever_line): the module or mechanism the lever executes, and whether
# it is served from the shared core's carried copy (core) or the kit's own tree (kit).
IMPL = {"bf16w": ("build_model.bf16w", "kit"), "trimul": ("opt_core.kernels.trimul", "core"), "trimul_exact": ("opt_core.kernels.trimul:exact+af3t_module", "core"), "triattn": ("opt_core.kernels.triattn", "core"), "transition": ("opt_core.kernels.transition:tier", "core"), "glu_proj": ("af3t_glu_proj.glu_proj", "kit"), "resid_fold": ("af3_kernels._residual_call", "kit"), "attn_epi": ("fpf_triatt_epi", "core"), "tmpl_trimul": ("opt_core.kernels.trimul", "core"), "pwa_lnl": ("lnl_fused.ln_linear", "kit"), "pwa_msa": ("lnl_fused.ln_linear", "kit"), "opm": ("af3t_opm", "kit"),
        "apb": ("opt_core.kernels.apb:tier", "core"), "stepgraph": ("diffusion_head.step_graph", "kit"), "hoist": ("diffusion_head.hoist", "kit"),
        "compile": ("torch.compile", "kit"), "sbatch": ("alphafold3._sample_diffusion_batched", "kit"), "atom_window": ("atom_window", "core"), "token_agg": ("af3t_token_agg.token_mean_relu", "kit"), "atom_rows": ("diffusion_transformer.forward_windowed_rows", "kit"), "prologue": ("diffusion_head.augment_and_noise_batched", "kit"), "dtk": ("dtk_kernels", "core"), "template_dedupe": ("template_dedupe.forward", "kit"), "dev_scalars": ("dev_scalars.install", "kit"), "tri_layout": ("tri_layout.forward", "kit"), "ln_rows": ("ln_rows.forward", "kit"), "attn_layout": ("attn_layout._attention", "kit"), "gate_fuse": ("gate_fuse.gated_", "kit"), "castcache": ("castcache.install", "kit"), "canonical_noise": ("canonical_noise._sample_diffusion", "kit"), "prefetch": ("pipeline_stream.await_batch", "kit"), "write_behind": ("pipeline_stream.await_results", "kit"), "autotune_cache": ("triton.knobs.autotuning.cache", "kit"), "feat_par": ("pipeline_stream.merge_featurise_reports", "kit"),
        "graph_drop": ("big.build_levers", "kit"), "diff_free": ("forward._reset_item_state", "kit"),
        "expandable_segments": ("opt_core.mem.torch_alloc", "core"), "prev_free": ("forward_impl._run_trunk_prev_free", "kit"),
        "paircond_chunk": ("paircond_rows.pair_conditioning_rows", "kit"),
        "rowpair": ("opt_core.mem.rowpair", "core")}
# The canonical cross-engine strategy id (the shared core's STRATEGIES.json) each lever is an instance of: the LEVER line's name=; the kit's own
# lever name rides beside it as lever=.
LEVERS_OFF = "levers_off"                                     # the LEVER line's reason= for a lever MODEL_OPT_LEVERS_OFF dropped from the mode for this run (modes.ENV_LEVERS_OFF; report.lever_line)
STRATEGY = {"bf16w": "F4.bf16_weight_precast", "trimul": "F2.fpf_trimul_fast", "trimul_exact": "F2.fpf_trimul_exact", "triattn": "F1.flash_triatt", "transition": "LOCAL.fused_transition", "glu_proj": "LOCAL.af3_torch.glu_proj", "resid_fold": "F5.kernel_glue_gates", "attn_epi": "F5.kernel_glue_gates", "tmpl_trimul": "F2.fpf_trimul_fast", "pwa_lnl": "F5.fpf_msa_kernels", "pwa_msa": "F5.fpf_msa_kernels", "opm": "F5.fpf_msa_kernels",
            "apb": "F5.flash_attn_dense", "stepgraph": "F3.cuda_graph_sampler", "hoist": "LOCAL.step_invariant_hoist",
            "compile": "F3.torch_compile", "sbatch": "LOCAL.af3_torch.sample_batch", "atom_window": "LOCAL.af3_torch.atom_window", "token_agg": "LOCAL.af3_torch.token_agg", "atom_rows": "LOCAL.af3_torch.atom_rows", "prologue": "LOCAL.af3_torch.batched_prologue", "dtk": "LOCAL.dit_fused_kernels", "template_dedupe": "LOCAL.af3_torch.template_distinct_slots", "dev_scalars": "F6.host_sync_elimination", "tri_layout": "F2.trimul_tperm_exact", "ln_rows": "LOCAL.af3_torch.ln_rows", "attn_layout": "LOCAL.af3_torch.attn_layout", "gate_fuse": "LOCAL.af3_torch.gate_fuse", "castcache": "LOCAL.af3_torch.castcache", "canonical_noise": "LOCAL.af3_torch.canonical_noise", "prefetch": "F6.item_ordering_prefetch", "write_behind": "F6.output_overlap", "autotune_cache": "F3.jit_cache_keyed", "feat_par": "F6.item_ordering_prefetch",
            "graph_drop": "F7.graph_pool_budget", "diff_free": "F7.chunked_eval",
            "expandable_segments": "F7.expandable_segments", "prev_free": "F7.chunked_eval",
            "paircond_chunk": "F7.chunked_eval",
            "rowpair": "F7.tensor_parallel"}

PORT = {   # the port itself (what makes xfold @ 22bdeed run the OpenFold3 parameter layout): install-time in the kit's copy, not a switch
    "files": ["af3_torch/xfold/of3.py", "af3_torch/xfold/nn/fourier_constants.py", "af3_torch/xfold/alphafold3.py", "af3_torch/xfold/params.py",
              "af3_torch/xfold/nn/attention.py", "af3_torch/xfold/nn/atom_cross_attention.py", "af3_torch/xfold/nn/diffusion_head.py",
              "af3_torch/xfold/nn/diffusion_transformer.py"],
    "what": "OF3 weight-layout flag (xfold.of3.OF3 = True), the OF3 haiku-record loader (params.py import_jax_weights_), seven sokrypton-fork "
            "code-path mirrors, two upstream fixes (recycles+1, Gaussian noise), the diffusion-head hoist + whole-step graph hooks",
}


# The kit's expected per-call fallbacks on the H100 stack: the kernels' own DECLARED coverage gates, served by the stock path by design and
# named in the kit's census (af3t/kernels/af3_kernels.py). Two kinds. Shape gates, item-independent, listed as exact census keys:
# `fallback:c=64` (af3_kernels.py `c=%d`: the trimul kernel covers c=128/256; the template embedder's c=64 pair stack is stock's) and
# `fallback:c=384,stock-row` (the transition provider measured the engine's own module fastest for a c=384 cell: the stock statement serves, by the table) and
# `fallback:c=64,no_cell` (a c=64 transition shape the provider has no cell for on the card: the stock statement, by name; listed so a card without cells is named, not judged).
# glu_proj: `fallback:c=384` (the single transition: stock), `fallback:c=128,rows<16384` / `fallback:c=64,rows<16384` (below the row floor: stock), `fallback:superseded:transition`
# (lever transition selected too: its kernel serves the module, glu_proj steps aside per call by that name).
# trimul_exact: `fallback:unvouched` (a call whose stack, width or size the shared core provider's table does not vouch its exact row for -- every call on
# an A100, items above the vouched sizes on an H100: the module's own statement, by name) and `fallback:superseded:trimul` (fast / big: lever trimul's
# tier word serves the class, trimul_exact steps aside per call by that name).
# Size gates, listed with SIZE_GATE in place of the threshold the kernel prints in its own census word: `fallback:N<101` (the shared
# core's fpf_trimul_v4 generic.N_MIN: the v4 row's floor; the provider's cells name other rows below it, so the key stays declared and unused) and triattn's `fallback:N<16`
# (af3_kernels.py) — such a key is expected ON AN ITEM iff that item's padded token count (its bucket, the N every trunk kernel call of
# the item sees) is below the printed threshold, so the expectation is derived per item from the declared gate and the item's size, never
# a hand count: every trimul call of a 24-token item (bucket 64) takes `N<101` by design; one such event on a 448-token item is
# degradation. Observed keys outside this table, a size key on an item the gate does not cover, a kernel_error count, calls no item
# accounts for, or a dead lever are degradation: the PARTIAL gate (cli.pred, fallback_account); the run record's fallbacks_expected keeps the
# expected ones, counted. A new expected key is a change to this table alone — never absorbed silently.
SIZE_GATE = "{lt}"
ARCH_GATE = "{sm}"                                     # an arch step-aside word: `fallback:arch=sm_103` names the compute capability a kit kernel carries no cells for
ARCH_CELL_MAJORS = {"opm": (8, 9)}                    # the capability majors each arch-gated kit kernel IS measured on (af3t_msa.OPM_CELLS): a step-aside naming one of these is NOT expected
ROW_WORD = "{row}"                                     # a provider's STOCK row answered by number (the cell's measured winner is a library / torch statement of the op): the module's own
_ROW_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.@:+-")   # statement serves BY NAME and the census names the row (row names: [A-Za-z0-9_.@:+-]+)
EXPECTED_FALLBACKS = {"trimul": ("fallback:c=64", "fallback:N<" + SIZE_GATE, "fallback:stock_row:" + ROW_WORD), "tmpl_trimul": ("fallback:stock_row:" + ROW_WORD,),
                      "triattn": ("fallback:N<" + SIZE_GATE, "fallback:refused:" + ROW_WORD + ":stock_row"),
                      "transition": ("fallback:c=384,stock-row", "fallback:c=128,stock-row", "fallback:c=64,stock-row", "fallback:c=64,no_cell"),
                      "glu_proj": ("fallback:c=384", "fallback:c=128,rows<16384", "fallback:c=64,rows<16384", "fallback:c=64,card-off", "fallback:superseded:transition"),
                      "trimul_exact": ("fallback:unvouched", "fallback:superseded:trimul", "fallback:stock_row:" + ROW_WORD), "opm": ("fallback:arch=sm_" + ARCH_GATE,)}


# The determinism class per mode: `same_position` — an (input, seed) at the same position of a pass is content-equal (every output file byte for
# byte, the fork's mmCIF timestamp line excepted) across two fresh model processes on one box; `position_independent` — an item's outputs are
# content-equal whether it runs first or second in a pass; `fresh_kernel_cache` — the same with an empty kernel cache. fast / big compile and
# autotune their Triton kernels per padded length per process, and the selected tiles can differ between processes (differences of the bf16
# class's own size), so they claim none of the three (test_modes_lock locks this table's shape).
DETERMINISM = {"off": {"same_position": True, "position_independent": True, "fresh_kernel_cache": True},
               "exact": {"same_position": True, "position_independent": True, "fresh_kernel_cache": True},
               "fast": {"same_position": False, "position_independent": False, "fresh_kernel_cache": False},
               "big": {"same_position": False, "position_independent": False, "fresh_kernel_cache": False}}
def size_gate_threshold(lever: str, key: str):
    """The threshold a size-gated census key prints (`fallback:N<101` -> 101) when EXPECTED_FALLBACKS lists that gate for the lever, else None."""
    for pat in EXPECTED_FALLBACKS.get(lever, ()):
        if SIZE_GATE not in pat:
            continue
        head, tail = pat.split(SIZE_GATE, 1)
        if key.startswith(head) and key.endswith(tail) and len(key) > len(head) + len(tail):
            num = key[len(head):len(key) - len(tail)] if tail else key[len(head):]
            if num.isdigit():
                return int(num)
    return None


def arch_gate_word(lever: str, key: str):
    """The capability an arch step-aside census key names (`fallback:arch=sm_103` -> 'sm_103') when EXPECTED_FALLBACKS lists the arch gate for the
    lever AND the capability's major is not one the kernel is measured on (ARCH_CELL_MAJORS: there the gate cannot fire, so such a word stays
    unexpected), else None. Item-independent: the card decides, not the item's size."""
    for pat in EXPECTED_FALLBACKS.get(lever, ()):
        if ARCH_GATE not in pat:
            continue
        head, tail = pat.split(ARCH_GATE, 1)
        if key.startswith(head) and key.endswith(tail) and len(key) > len(head) + len(tail):
            digits = key[len(head):len(key) - len(tail)] if tail else key[len(head):]
            if digits.isdigit() and len(digits) >= 2 and int(digits[:-1]) not in ARCH_CELL_MAJORS.get(lever, ()):
                return "sm_" + digits
    return None


def stock_row_word(lever: str, key: str):
    """The row a stock-row census key names (`fallback:stock_row:torch_math` -> 'torch_math', `fallback:refused:sdpa:stock_row` -> 'sdpa') when
    EXPECTED_FALLBACKS lists a stock-row pattern (ROW_WORD) for the lever and the row spells as a row name ([A-Za-z0-9_.@:+-]+), else None: the
    shared core's provider answered a STOCK row by number for the cell, so the module's own statement served by name -- correct outputs, a named
    aside. Item-independent. A named refusal of a KERNEL row (`fallback:refused:<row>:<RefusalKind>`), a bare `fallback:refused`, a kernel_error
    or any other word matches no pattern here and stays unexpected."""
    for pat in EXPECTED_FALLBACKS.get(lever, ()):
        if ROW_WORD not in pat:
            continue
        head, tail = pat.split(ROW_WORD, 1)
        if key.startswith(head) and key.endswith(tail) and len(key) > len(head) + len(tail):
            row = key[len(head):len(key) - len(tail)] if tail else key[len(head):]
            if row and all(ch in _ROW_CHARS for ch in row):
                return row
    return None


def fallback_expected(lever: str, key: str, bucket=None) -> bool:
    """Is this census key expected for the lever — an exact shape-gate key, an arch step-aside on a capability the kernel carries no cells for
    (arch_gate_word), a provider's stock row answered by number (stock_row_word), or a size-gate key on an item whose padded token count
    (`bucket`) is below the threshold the kernel printed (None = not attributable to an item: a size gate is then NOT expected)."""
    if key in EXPECTED_FALLBACKS.get(lever, ()):
        return True
    if arch_gate_word(lever, key) is not None or stock_row_word(lever, key) is not None:
        return True
    t = size_gate_threshold(lever, key)
    return t is not None and bucket is not None and int(bucket) < t


def fallback_split(events: dict, bucket=None) -> tuple:
    """(expected, unexpected): fallback / kernel_error counters {lever: {key: n}} split by fallback_expected at one padded token count
    (`bucket`; None = the pass as a whole, where only the shape gates are expected)."""
    exp, unexp = {}, {}
    for lever, counts in (events or {}).items():
        for k, v in counts.items():
            (exp if fallback_expected(lever, k, bucket) else unexp).setdefault(lever, {})[k] = v
    return exp, unexp


def fallback_account(records, events: dict) -> dict:
    """The pass's fallback accounting, per item: every (input, seed) forward record's own counters (`fallbacks`, the census increase over
    the item; `bucket`, its padded token count) are split by fallback_expected at that item's size; calls the pass census (`events`, the
    model process's totals) holds beyond what the items account for are split with no size (shape gates only). Returns {expected,
    unexpected} ({lever: {key: n}}, summed) and per lever {observed: every fallback / kernel_error call, accounted: the expected ones,
    size_gated: {key: [item names]} — the items on which a size gate was expected}."""
    exp, unexp, levers, seen = {}, {}, {}, {}
    def add(dst, lever, k, v):
        dst.setdefault(lever, {}); dst[lever][k] = dst[lever].get(k, 0) + v
    for rec in records or ():
        for lever, counts in (rec.get("fallbacks") or {}).items():
            for k, v in counts.items():
                v = int(v); add(seen, lever, k, v)
                lv = levers.setdefault(lever, {"observed": 0, "accounted": 0, "size_gated": {}})
                lv["observed"] += v
                if fallback_expected(lever, k, rec.get("bucket")):
                    add(exp, lever, k, v); lv["accounted"] += v
                    if size_gate_threshold(lever, k) is not None and rec.get("name") not in lv["size_gated"].setdefault(k, []):
                        lv["size_gated"][k].append(rec.get("name"))
                else:
                    add(unexp, lever, k, v)
    for lever, counts in (events or {}).items():                              # the census beyond the items: build-time or unattributed calls
        for k, v in counts.items():
            rest = int(v) - seen.get(lever, {}).get(k, 0)
            if rest <= 0:
                continue
            lv = levers.setdefault(lever, {"observed": 0, "accounted": 0, "size_gated": {}})
            lv["observed"] += rest
            if fallback_expected(lever, k, None):
                add(exp, lever, k, rest); lv["accounted"] += rest
            else:
                add(unexp, lever, k, rest)
    return {"expected": exp, "unexpected": unexp, "levers": levers}


# The memory levers of `big` (big.py LEVER_ORDER; family F7): composed in the package on the shared core's opt_core.mem, never in the
# carried kit. `class` = the byte-equality property each lever carries by construction or by measurement; the composition's own ceiling
# and walls are not tracked in this repo.
BIG_LEVERS = {
    "graph_drop": {"family": "F7", "kind": "setting", "class": "the remaining levers' (graph replay is the same kernels)", "name": "fast's step graph switched off per item at or above big.GRAPH_DROP_MIN_TOKENS padded tokens (the hoist kept, the step eager); replayed below it"},
    "diff_free": {"family": "F7", "kind": "setting", "class": "bitwise vs the mode without it (frees only)", "name": "diffusion statics released when the sampler returns, before the heads"},
    "expandable_segments": {"family": "F7", "kind": "allocator", "class": "bitwise (placement only)", "name": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True via opt_core.mem.torch_alloc (declared beside the step graph below graph_drop's gate)"},
    "prev_free": {"family": "F7", "kind": "setting", "class": "bitwise vs the mode without it (frees only)", "name": "the recycle's fp32 prev embeddings released once the Evoformer has embedded them"},
    "paircond_chunk": {"family": "F7", "kind": "setting", "class": "tolerance (per-pair arithmetic unchanged; GEMM leading size per block)", "name": "the sampler's step-invariant pair conditioning evaluated per block of big.PAIRCOND_CHUNK_ROWS rows into one preallocated result (xfold/nn/paircond_rows.py)"},
}

# The resource axis of `big`: `--n_gpu P` (served set modes.N_GPU_SUPPORTED = (1, 2, 4, 8)). P = 1 is the single-GPU composition;
# P > 1 row-shards the pair representation over P GPUs of one box on the shared core's opt_core.mem.rowpair, end to end (forward.py
# main_sharded spawns the ranks through the core's launcher; rowpair_xfold.py drives the core's row-sharded seams with the xfold modules'
# own layers as callables — no class is rebound). ONE lever line per pred names it (report.rowpair_line): what the ranks ran, from rank 0's
# forward.json. The kit statements rowpair_xfold.py restates on a row shard and forward_impl._run_trunk_prev_free mirrors are named,
# one gate word each, in forward.ROWPAIR_GATES.
N_GPU_LEVER = {
    "rowpair": {"family": "F7", "kind": "sharding", "impl": "opt_core.mem.rowpair", "scheme": "rowpair",
                "class": "fast's, as a band (the TriMul seam is the core's row schedule: fpf_v4 row-block kernels at pair sizes >= its min_tokens, the eager composition below; row-count-dependent GEMM tiles)",
                "sites": "trunk + MSA-stack pair ops + template pair stack + the single track's pair logits, row-sharded; diffusion sampler, confidence head and distogram head dense on rank 0 (the P > 1 ceiling term)",
                "name": "row-sharded pair stack over P GPUs (big --n_gpu P)"},
}


# Card classes per lever (opt_core.arch — ONE declaration each; the card words (card_table below, opt_core.arch.card_table / lever_state) render
# from these rows, never typed; the row keys are opt_core.arch.declare's keywords). The first key = the sm classes a lever was tested on: H100 sm90
# and A100 sm80 for every lever of every mode — `exact`, `fast`, `big` and `big --n_gpu P` (ARCH_TESTED; on the A100 the levers run on the
# the shared core providers' `8.0` cells: triangle attention, TriMul, transition, pair-bias attention); min_sm = the floor the
# implementation needs (bf16 autocast and the Triton kernels: sm80+); B200 sm100 / B300 sm103: no test with this tree (the word
# `uncertified` in the table: the levers engage where the shared core's providers serve a cell for the card and step aside by name elsewhere, saying so).
ARCH_TESTED = ("sm80", "sm90")
ARCH = {lever: {"certified": ARCH_TESTED, "min_sm": "sm80"} for lever in (*LEVERS, *BIG_LEVERS, *N_GPU_LEVER)}   # the row keys are opt_core.arch.declare's own keyword names (declare_arch passes each row as **row)


def arch_lever_id(lever: str) -> str:
    """The id a lever of this kit is declared under in opt_core.arch (one blank-free token, prefixed with this engine's name so ids never collide)."""
    return f"af3_torch.{lever}"


def declare_arch():
    """Declare every lever's card classes in opt_core.arch (idempotent: the same content twice is a no-op) and return {lever: Support}."""
    from opt_core import arch
    return {lever: arch.declare(arch_lever_id(lever), **row) for lever, row in ARCH.items()}


def card_table(sms=None):
    """{lever: {sm: word}} — opt_core.arch.card_table over this kit's levers."""
    from opt_core import arch
    declare_arch()
    t = arch.card_table([arch_lever_id(l) for l in ARCH], sms)
    return {lever: t[arch_lever_id(lever)] for lever in ARCH}

