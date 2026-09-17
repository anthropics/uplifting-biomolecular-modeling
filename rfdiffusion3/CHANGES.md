# RFdiffusion3 kit — what changes vs stock

Stock = RFdiffusion3 at the pin (STOCK.md), never edited. The kit is a ten-file overlay on the installed `rfd3/model` package of the KIT
interpreter (`opt/forward/xattempt_addon/patched/rfd3/model/`: eight upstream files with guarded hunks and two new modules, `hoist.py`
and `cudagraph_sampler.py`, applied by `python -m rfdiffusion3_opt.install apply`) plus the package `opt/rfdiffusion3_opt`, which gates
a mode and exports its switches before anything imports `rfd3.model`. Each lever is a branch in those files keyed on one `RFD3_*` switch
read once at import or at sampler construction; with every switch unset the files take upstream's code path, and `off` runs the pristine
interpreter, where none of them exist. A mode is all of its levers; `fast` includes `exact`'s. Lever names are the switch names printed on
the run's `ACTIVE … env=` and `APPLIED …` lines; `opt/rfdiffusion3_opt/modes.py` is the one mode table. File names below are relative to
the overlay's `patched/rfd3/model/`; its `README.md`, `KNOWN_ISSUES.md` and `docs/INVARIANCE_NOTES.md` hold the per-file detail.

## exact — outputs identical to stock

Final coordinates and sequence indices bit-identical to eager stock at the same seed and `diffusion_batch_size`, under `--det 1`.

- `RFD3_HOIST` — roll-out cache: step-invariant tensors — the atom-pair bias `to_b(P_LL)` of the 3 atom-encoder + 3 atom-decoder
  attention blocks, `downcast_c(C_L, S_I)`, the dense `(B, L, L)` key mask, the masked bias written in one `torch.where(…, out=)` pass,
  the decoder blocks' masked bias across the second recycle — are computed once per roll-out or per denoiser call by the unchanged
  upstream modules and reused (`hoist.py`, `RFD3_diffusion_module.py`, `layers/attention.py`, `inference_sampler.py`). Numerics:
  bitwise (caching only). Steps aside: never.
- `RFD3_FZT` — fused SwiGLU transition: `Transition._forward_impl` for the (c, n·c) cells (128,256), (128,512), (256,512), (256,1024)
  on CUDA under bf16 autocast in eval mode is served by the shared core's FlashPairformer transition kernel `opt_core.attn.pair_fused.transition` (Triton,
  impl `fpf`, `ln='stock'`: the module's own RMSNorm output in, stock's bf16 rounding points kept, the n·c hidden never written to memory;
  `layers/layer_utils.py`). Numerics: bitwise on the pinned stack. Steps aside, by name: on compute capability 8.0, row counts outside
  `FZT_ROWS_BY_CC` (`128x256` below 16384 rows, the other cells below 4096, any cell above 1048544) run the stock body — counted
  `cells_stock['rows<MIN:<cell>']` / `['rows>MAX:<cell>']`, named `rows_cc=8.0 rows_served=…` on the `APPLIED … layer_utils` line, by
  design; a cell the core's table does not serve on this compute capability / Triton runs the stock body under one
  `FALLBACK lever=RFD3_FZT reason=no-cell …` line (a partial activation, exit 3).
- `RFD3_CUDAGRAPH` — CUDA-graph sampler: the denoiser step is captured once per roll-out signature (batch size, tensor shapes,
  `n_recycle`) and replayed for that roll-out's calls and for later design batches with the same signature; everything the stock loop
  does outside the denoiser call — origin jitter (`s_jitter_origin`), per-step re-centring and random rotation / translation around the
  motif (`allow_realignment`, `fraction_of_steps_to_fix_motif`), the closing motif re-insertion and alignment, a spec's `partial_t` —
  runs from upstream's own statements with the Gaussian draws in stock order (`cudagraph_sampler.py`; `[RFD3_CUDAGRAPH]` hunks in
  `layers/*.py` and `RFD3_diffusion_module.py`). Numerics: bitwise (the eager step's kernels replayed on the same bytes). Steps aside:
  a failed capture is served by the eager denoiser and counted — `fallbacks` > 0 is a partial activation, exit 3, and
  `LEVER name=RFD3_CUDAGRAPH state=on|skipped [reason=…]` says which; an out-of-memory error at capture is re-raised, never rerouted.
- `RFD3_INIT_CHUNK` — row-blocked initialiser embedding: the token initialiser's two all-atom-pair sinusoidal embeddings
  (`SinusoidalDistEmbed`, once per design batch) run over row blocks of at most 2²⁰ atom pairs, so their fp32 intermediates stay
  bounded instead of growing with L_atom² (`layers/blocks.py` → `layers/layer_utils.py`). Numerics: bitwise (the same statements per
  block). Steps aside: never (an item of up to 1024 atoms is stock's single block).

Atom-attention path under `exact` and `fast`: upstream's free-memory rule (dense or sparse atom attention per call) is evaluated once per
attention shape at the first denoising step, printed (`ATTN_PREFLIGHT … pinned=1`) and held for the process; a shape it sends sparse is
refused before the first step (`NOT ACTIVE: KERNELS refused …`, exit 5: lower `diffusion_batch_size`, or `--mode off`).

## fast — within stock's seed-to-seed variation

- `RFD3_COMPILE` — upstream's own four compile targets (encoder, diffusion_token_encoder, diffusion_transformer, decoder) wrapped in
  `torch.compile` (inductor, automatic dynamic shapes) once per process at the first roll-out through the shared core's
  `opt_core.capture.compile`; the cache accessors, the graph sampler's memo helpers, the atom-attention path decision, the index
  builder and the fused transition stay eager as dynamo boundaries, and the compiled step is what the graph captures (`hoist.py`).
  Numerics: inductor re-association. Steps aside: fail-closed — fewer than 4/4 targets wrapped, no compiled frame run or a
  recompile-limit hit reads `compile=fallback:kit-compile-gate(…)` on the `KERNELS` line, exit 5; a `frames_skipped` count other than
  the pinned stack's is named on a `COMPILE frames_skipped=…` line and the lever stays engaged.
- `RFD3_TOKEN_SDPA` — the 18-block token transformer's pair-bias attention runs through upstream's own `dense_sdpa_pairbias_attention`
  (`F.scaled_dot_product_attention` of torch 2.13 over the same key set and `-inf` mask; `layers/attention.py`). Numerics: SDPA's
  summation order. Steps aside: never on CUDA inference.
- `RFD3_GATHER_ATTN` — the atom-level index-set attention (`LocalAttentionPairBias`, 3 atom-encoder + 3 atom-decoder blocks; per query
  atom k = 128 listed atoms, pair bias, sigmoid gate) is served by the shared core's `opt_core.kernels.gather_attn` (Triton, compiled
  for the running card, sm_90 or sm_80): K / V rows and the k bias scalars gathered straight from `to_b(P_LL)` in its `(L, L, H)` layout,
  fp32 online softmax, bf16 P·V with fp32 accumulation; no `(D, H, L, L)` bias is built, so upstream's dense-memory rule and sparse
  fallback do not arise and `fast` has no atom-attention refusal. Installed at the engine's `initialize()` by wrapping two names of the
  overlaid attention module (`opt/rfdiffusion3_opt/gather.py`). Numerics: fused-kernel rounding, run-to-run bitwise (fixed loop order,
  no atomics). Steps aside: a call the kernel refuses by name (cell, dtype, layout, device) is served by the stock dense body and counted
  — `atom_attn=partial:gather_attn(…)` on the `KERNELS` line, exit 5.

## Every mode

- No `big` mode and no multi-GPU option: a kit mode runs the `diffusion_batch_size` it is given.
- Refused by name before anything runs (`NOT ACTIVE: mode=<m> cannot serve …`, exit 3; `--mode off` runs them as stock): classifier-free
  guidance (`inference_sampler.use_classifier_free_guidance=true`: a second, unconditioned pass per step, while the cache and the captured
  step hold one conditioning per roll-out), `low_memory_mode=true` (no resident all-atom pair track), the symmetry sampler
  (`inference_sampler.kind=symmetry`: its own per-step loop) and upstream's `compile_model=true`; a spelling the line reader cannot see is
  refused with the same line at the engine's initialize. The CFG parameters alone (`cfg_scale`, `cfg_t_max`, `cfg_features`) are served.
- Also `NOT ACTIVE`, exit 3: the shared core absent or below the `opt/pyproject.toml` floor; the overlay's files not its `MANIFEST.json`
  listing; this interpreter's `rfd3/model` files not in the overlay's state; no visible GPU; a kit switch preset at another value or an
  upstream attention switch (`RFD3_DENSE_SDPA_ATTENTION`, `RFD3_LOW_MEMORY_MODE`) set; `rfd3.model` imported before activation. A planned
  lever without evidence of application is a partial activation (`NOT ACTIVE: partial activation — <lever (reason)>`, exit 3).
- `KERNELS` census: one line per pass on every route naming the accelerator paths the pass ran (`compile=`, `atom_attn=`, `rmsnorm=`, …
  `ckpt=<basename>@<sha256[:12]>`), each word read from the object bound at upstream's call site, expected kinds in `routes.py`; under
  `exact` / `fast` a word that misses its kind ends the pass (`NOT ACTIVE: KERNELS refused …`, exit 5), under `off` it is reported only.
- The environment — GPU class, torch, Triton, the upstream checkout, a kernel cell in the core's `candidate` service — is named on the
  lines (`gpu=`, `torch=`, `PINS …`) and every lever engages; no mode runs under its name with a subset of its levers. A100
  (`configs/a100.env`): the levers read the card from the device and every lever engages on compute capability 8.0 (the fused transition
  within its 8.0 row range as above, the gather kernel and the compiled denoiser on their sm_80 code); the H200 (`configs/h200.env`,
  compute capability 9.0) runs every lever as the H100; no other card is configured.
- No stock exceptions (STOCK.md) and no opt-in upstream fixes: `off` and the kit modes run the pin as shipped.

## Switches

- `--det 0|1` (`RFDIFFUSION3_OPT_DET=1` on upstream's own command line) — the deterministic recipe (`opt/rfdiffusion3_opt/det.py`:
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=True)`, cuDNN deterministic, and
  `foundry.utils.torch.scatter_mean` served by the fixed-order segment reduction of `det_scatter.py`), identical on every mode, the
  stock child included; default `0` = production numerics, under which upstream itself is not bitwise run to run.
- `--allow-partial` (`RFDIFFUSION3_OPT_ALLOW_PARTIAL=1`) — records a partial activation in `opt_manifest.json` (`PARTIAL allowed: …`)
  and keeps the run's own exit code instead of exit 3. Refusals by name have no allowance.
- A mode is selected whole: `--mode`, or `RFDIFFUSION3_OPT=<mode>` on upstream's unchanged command line through
  `rfdiffusion3_opt_autoload.pth`. Variables and defaults: STOCK.md.
- Overlay management: `python -m rfdiffusion3_opt.install status|classify|rebind|apply|check-only|uninstall` (`uninstall` restores the
  `<file>.hoist_orig` originals; `rebind` re-bases a pristine or earlier install of this overlay; `apply` refuses any other state, exit 4).
