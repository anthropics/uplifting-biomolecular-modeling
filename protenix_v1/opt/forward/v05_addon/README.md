# v05_addon — the kit's lever code for Protenix 1.x (protenix==1.1.0)

Adapters only (no weights, no kernels: the Triton / CUDA kernels the levers launch are the shared core's, `common/opt_core`, served by name).
`protenix_v1_opt.kit` puts `lib/`, `lib/kit112_src/` and `ptxfpf/` on `sys.path` (in that order) when a mode activates. What each lever
replaces and its numerics class: `protenix_v1/CHANGES.md`.

## Layout

| path | what |
|---|---|
| `ptxfpf/levers_ptx1.py` | the lever adapter: run-time patches over the pip-installed protenix 1.1.0 (`bind_model`, `apply(arm)`, `describe`). Arm grammar `<trimul>[+lever...]`, trimul in stock / fast / exact, levers = `LEVER_NAMES`; the triangle-multiplication, triangle-attention and transition routes and the sampler-graph / hoist wiring live here. |
| `ptxfpf/ptx1_templ.py` | the template-embedder levers (`template_dedupe`, `tmpl_*`). |
| `ptxfpf/apb_ptx1.py` | the sampler attention-with-pair-bias levers (`ditattn`, `ditattnfp16`, `atomattn`) over the shared core's `opt_core.kernels.apb` rows. |
| `ptxfpf/trunk2_ptx1.py` | the trunk-side fused levers (`pfattn`, `opm_fused`, `pwa_fused`) over `opt_core.kernels.apb` and `lib/protenix_fpf_msa`. |
| `ptxfpf/ditfast_ptx1.py` | the fused diffusion-sampler levers (`cond_dedupe`, `dit_fused`, `dit_lowp`, `atom_fused`) over `lib/protenix_fpf_ditfast`, and `atom_attn_exact` over `opt_core.kernels.apb`'s `exact` tier word. |
| `lib/kit112_src/` | the CUDA-graph lever and the hoist: `infopt_graphs` (GraphedFunction, the graphed denoiser step `sg`, its host path `sampler_prep`, the stream-correct fast-LayerNorm rebuild), `dit_hoist` (lever `hoist`: the DiT step-invariant tensors computed once per sampler call). |
| `lib/ptx1_lazy_init.py` | the `lazy_init` lever: the model's dead random initialisation skipped during the stock runner's construction (the strict checkpoint load overwrites it); installed by the package before the runner is built. |
| `lib/ptx1_summary_host.py` | the `summary_hostidx` lever: `sample_confidence.compute_full_data_and_summary` with the chain / chain-pair bookkeeping from one host copy of the ids (EXACT-bitwise). |
| `lib/ptx1_keep_pool.py` | the `keep_pool` lever: the caching-allocator policy (the stock in-forward `torch.cuda.empty_cache()` sites of the confidence head become counted no-ops; the runner's per-item release and the kit's own releases pass through). |
| `lib/fastln_prebuilt*` | the prebuilt stream-correct fast-LayerNorm extension for the pinned stack (torch 2.13.0+cu130) and its loader; any other stack rebuilds it from stock's sources once per machine under `$TORCH_EXTENSIONS_DIR/fastln_stream` (checked bit-equal to the original before it is used either way). |
| `lib/detpatch/` | the deterministic scatter modules `--det 1` loads in place of `protenix.utils.scatter_utils` (`protenix_v1_opt/det.py`). |
| `lib/protenix_fpf_ditfast/` | the fused diffusion-sampler package: `cond_dedupe` (DiffusionConditioning once per step on the sample-invariant noise level), `dit_fast` (the 24-block token DiffusionTransformer as one fused forward around lever `ditattn`'s kernel; `PTX_DIT_LOWP` its fp16 word), `atom_fast` (both atom transformers as fused stacks around `atomattn`'s kernel); row kernels from `opt_core.kernels.apb.ditfast`. |
| `lib/protenix_fpf_msa/` | the MSA-module lever package (`opm_fused`, `pwa_fused`): cells and load check; kernels from `opt_core.ops.msa_fused`. |
| `inputs/` | the `warm` verb's input: the public complex `p995_1brs.json` (1BRS barnase ×5 + barstar ×5, 995 tokens) and its MSAs (`msa/`). |
