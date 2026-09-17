# Evo 2 kit — what changes vs stock

Stock = Evo 2 0.6.0 on vtx 1.1.0 at the pin (STOCK.md). Each lever is a patch the kit installs over one `vortex` /
`transformer_engine` / `evo2` function or class attribute when a model is constructed under a kit mode; `off` loads none (its
construction: STOCK.md §How stock is run). A mode is
all of its levers or a refusal by name (`KitRefused` out of the constructor when a launch condition — block counts, hidden size, the
`Wqkv` layout, kernel / FP8 flags — does not hold). Lever names are the ones printed on the `[evo2-kit] APPLIED … levers=N (…)` line;
stock sites cite `stock/src/`. The route is read from the constructed model: `vortex-kernels` (`Evo2(name, use_kernels=True)`) and
`torch-conv` (`Evo2(name)`), both with Transformer Engine's FP8 input projections ("TE routes" below), or `bf16-projections`
(Transformer Engine absent: upstream's bf16 fallback, `evo2_7b` only). All kernels are the kit's own Triton kernels plus direct cuFFT /
cuBLASLt calls into the libraries torch loads; the shared core is not imported. Code: `opt/evo2_opt/` (`activation.py`, `kit/`,
`gen/`) and `route/evo2_route.py`; `kit/gemm.py` groups the projection-side levers under the short keys L2, W1, E50, E56, E61.

## exact — outputs identical to stock

Numerics, every lever: the same operations in the same order per output element on the same weight layout (caching, placement and
fusion without re-association), so logits, scores, per-step generation logits and sampled tokens equal stock's on the same stack.

- `L1b_hcl_filter_cache` (torch-conv, bf16) — `HyenaCascade.compute_filter` (`vortex/model/model.py:400`) cached per (layer, length)
  after the first call, built per 512-channel tile with the stock expression; the prefill's time-vector update is kept.
- `L1c_hcm_filter_spectrum_cache`, `L1d_hcl_filter_spectrum_cache` (TE routes; L1d vortex-kernels) — the filters' `rfft` (`engine.py:46`,
  `:306`; `hcl_interface.py:261`, `hcm_interface.py:290`) cached per (layer, length, device), one length resident at a time. Steps
  aside: on vortex-kernels a layer whose spectrum exceeds the budget (1/7 of device memory) transforms per call as stock does, named.
- `E10_persistent_padded_fft_buffer` (TE routes) — the zero-padded fp32 FFT input is one persistent buffer per shape whose tail is
  zero by allocation; `x1*v` is cast into its head by one Triton launch.
- `E7_triton_fir3_featurizer` (all), `E8_bf16_depthwise_fir`, `E9a_hcl_fftconv_epilogue`, `E9b_hcm_fftconv_epilogue` (torch-conv) —
  Triton kernels for the 3-tap featurizer FIR and the 7-tap short FIR fused with the `x1*v` / `x2` gates and casts of `parallel_fir` /
  `parallel_iir` (`engine.py:162`, `:306`), rounding to bf16 at every stock op boundary.
- `E7k_featurizer_fir_inline_hcl_hcm`, `E9k_strided_conv_epilogues`, `E62_fused_hcs_gate_conv` (vortex-kernels) — the 3-tap FIR is
  applied inline by the pad and epilogue launches, so the featurizer output is never written; the epilogues (`engine.py:171-275`)
  read the inverse transform at its own strides; `hcs_conv` (`hcs_interface.py:117`) runs gate, fp32 depthwise FIR, cast and gate as
  one kernel in stock tap order. Steps aside: generation state, padding masks or a skip bias take the stock call.
- `E59E60_direct_cufft_plans`, `C2R_OUT_c2r_output_aliases_spectrum_storage` (TE routes), `C_SPEC_compact_half_spectrum` (torch-conv)
  — cuFFT called directly with torch 2.7.1's own plan parameters, one plan set per device, in place of `torch.fft.rfft` / `irfft`
  (whose C2R clones its input); one-sided spectra in persistent buffers, the product in place, the inverse written over dead spectrum
  storage.
- `E11_fused_rmsnorm_tail`, `E65_attention_residual_norm`, `E66_residual_into_next_pre_norm` (all) — one Triton launch computes
  `RMSNorm` (`layers.py:195`) with the L2 norm in ATen's reduction order for a contiguous bf16 row (torch 2.7.1 `Reduce.cuh` layout;
  `kit/rmsnorm_rule.py` picks the thread count ATen would) and the stock tail op by op; the attention block's residual + post_norm and
  each block's closing residual add are folded into the next norm launch with torch's bf16 add. Steps aside: a width outside the rule
  takes ATen's norm plus the fused tail, named once; padding masks, activation logging or a generation call take the stock forward.
- `E63_pre_norm_fp8_emit` (TE routes), `E64_channels_first_projection` (vortex-kernels) — the pre_norm launch writes Transformer
  Engine's e4m3 GEMM input with TE's own cast arithmetic and the module's input scale, replacing its cast-and-pad copy; the projection
  is TE's `general_gemm` on the same operands (E64 swaps operand roles — the same TN GEMM and K order — so channels come out contiguous
  along L). Steps aside: before a module's first FP8 call, and under autograd, the stock composition runs.
- `L2_interleave_folded_into_te_master_weight` (TE routes) / `L2_interleave_fold_bf16_projection_weight` (bf16) — the `interleave()`
  copy (`utils.py:26` at `model.py:262`) folded once, in place, into the projection weight rows and featurizer taps at apply; the GEMM
  sees the same numbers in the same reduction order.
- `W1_fp8_weight_workspace_cache`, `E61_general_gemm_out_buffer` (TE routes) — TE's FP8 cast of the bf16 master weight kept per module
  instead of redone per call; GEMM output buffers kept per (shape, dtype, device); a GEMM on the second device gets its own workspace.
- `E50_out_filter_dense_matmul_on_transposed_view`, `E56_fused_bias_residual` (all) — `out_filter_dense` + bias + residual
  (`model.py:541`, `:583`): the stock `matmul` on the transposed view (copy-free at batch 1), then bias + residual (+ post_norm) in one
  launch in stock rounding order.
- `R1_rotary_table_reuse` (TE routes) — the rotary cos / sin tables (`positional_embeddings.py:36`) reused under inference mode when the
  cached length covers the call (same device and dtype, no xPos scale) instead of rebuilt per forward.
- `T9_cublaslt_config_selection` (all) — for the MLP, attention and batch-1 `out_filter_dense` GEMMs with M ≥ 512, cuBLASLt candidates
  restricted to one sequential fp32 accumulation over k (no split-K) run on the live operands, are admitted only if their output equals
  torch's bit for bit, and are kept only if faster in an in-stream trial; at most 8 M values per device. Steps aside: a signature that
  ever differs retires to torch's call, named.
- `GATE_forward_gate_shape_manager`, `G1_forward_graph_replay` (TE routes; G1 one device) — `StripedHyena.forward` takes the kit path at
  every (batch, length); shape-bound buffers, plans and the graph follow the current shape. A recurring scoring shape of at most 16,384
  tokens is recorded into a CUDA graph during an eager call and replayed after (same kernels and buffers; a fresh clone of the logits).
  Steps aside: cached-generation calls, padded forwards, two-device models, autograd and hooked models (`return_embeddings=True`) run
  eagerly or on the stock path for that call, named once and counted on the EXIT line; a hook on a folded projection layer is refused.
- `S1_prepare_batch_numpy_rows` (all) — `prepare_batch` (`evo2/scoring.py:10`) builds the identical `input_ids` / `seq_lengths` as one
  numpy array. Steps aside: a tokenizer other than vortex's `CharLevelTokenizer` takes the stock function.
- Pipelined scoring, `pipeline=on(…)` on the APPLIED line (`evo2_40b` on two devices) — consecutive same-length windows of a
  `score_sequences(batch_size=1)` call run two-in-flight across vortex's layer split on per-device streams; scheduling only, each window
  is the stock one-row forward. Steps aside: `pipeline=off(…)` by name unless the installed `evo2/scoring.py` is the pinned file;
  batched calls run the stock batched forward unsplit. Multi-device guards (`kit/multidev.py`) launch every patched kernel with the
  current device set to its input's device and keep one cuFFT plan store per device; no-ops on one device.
- Generation member `hyenafuse` — the cached decode step's per-token Hyena state update (FIR state shift, IIR pole / residue recurrence,
  gates) in one Triton launch per block, stock arithmetic and rounding per element. Steps aside: no Hyena blocks → n/a by name; a
  prompt shorter than a block's FIR length → the stock step for that block until its state is full, counted.
- Generation member `cudagraph` — after the prefill and the first decode step, the one-token step is warmed up eagerly on a side stream,
  captured into a CUDA graph per device segment and replayed for every remaining token of the call. Steps aside: a call shape the
  capture cannot serve runs the stock eager step, named; a capture that raises is `CaptureRefused` by name. The prefill and every
  forward carrying inference params run the stock path.

## fast — within stock's seed-to-seed variation

Kit-true class: scoring and every forward identical to stock; sampled tokens identical in distribution (each token an exact sample from
the target's distribution; a seed's sequence differs from the stock sampler's).

- Every `exact` lever and member, unchanged.
- Generation member `specdec` — speculative sampling (Leviathan et al. 2023; Chen et al. 2023) inside `Evo2.generate` with the
  `evo2_1b_base` draft built by the stock constructor on the target's first device: k = 6 drafted tokens per round, one (1, k+1) cached
  target call judges them with the standard accept / residual-resample / bonus rule (float64 arithmetic, torch's generator); Hyena
  layers step the stock sequential update once per position and both models' states roll back to the kept position from per-position
  snapshots. Numerics: every emitted token is an exact sample from the target's `temperature` / `top_k` / `top_p` distribution of that
  call; the sequence for a seed is not the stock sampler's; greedy returns the target calls' argmax path; `GenerationOutput` carries
  the target-call logits row per emitted token. Steps aside: `cached_generation=False` or `force_prompt_threshold` → one FALLBACK line
  and the exact loop for that call; no draft beside the target's checkpoint or under `$EVO2_OPT_WEIGHTS` → `specdec=refused` and
  `generate()` raises `Evo2OptRefused` naming the install command; a draft off its pinned sha256 → one UNPINNED line, proceeds;
  Transformer Engine absent → `specdec=n/a`, `generate()` stays exact's.

## Every mode

- No stock exceptions (STOCK.md): `off` is upstream as installed with no setting added, and no upstream fix is shipped.
- Costs: apply is the in-place weight fold and class patching, no forward. The first forward of a (batch, length) fills that shape's
  caches and plans with the stock computation and compiles the Triton kernels on first launch; at a recurring shape the next forwards
  carry T9's trial and G1's capture. Caches are device memory beside the model, one length's caches and one shape's buffers resident;
  an allocation failure is torch's `OutOfMemoryError` from the call that made it and is never caught to reroute. A Triton launch past
  32-bit element addressing (`kit/i32.py`) is refused by name before the launch.

## Switches

- `EVO2_OPT=exact | fast | off` (unset = off), or `evo2_opt.enable(mode)` / `evo2_opt.status()` in the process; `EVO2_OPT_WEIGHTS` and
  `EVO2_OPT_HOME` (STOCK.md §Variables). There is no per-lever switch; the kit reads no other variable, flag or file and writes only its
  `[evo2-opt]` / `[evo2-kit]` / `[evo2-gen …]` lines on stderr.
