# Proteina-Complexa kit — what changes vs stock

Stock = Proteina-Complexa 1.1.0 at the pin (STOCK.md). Each lever is a replacement the kit installs over one stock method or module
function at start-up under a kit mode: the install's `complexa_opt_autoload.pth` imports `complexa_opt._autoload` at interpreter start
(inert without `COMPLEXA_OPT`), which arms a one-shot import hook on `proteinfoundation.proteina`; when upstream's generation process
imports its model module the hook installs the mode's levers, before the checkpoint loads. `off` loads none. Upstream's files, weights
and configuration are untouched on every mode. A mode is all of its levers: one that cannot be installed (an upstream class or statement
not as pinned) refuses the whole mode by name (`NOT ACTIVE`, exit 3); one that never engaged during the run makes it exit 3
(`KIT-RECORD … state=partial`). `fast` and `big` each add to `exact`'s four levers; they do not include each other's. Lever names are
the ones printed on the run's `LEVER name=…` lines. The kit binds no compiled kernels: every lever is PyTorch over upstream's own modules;
the shared core (`common/opt_core`) supplies the mode table, the start-up hook, the stderr line grammar and the manifest.

## exact — outputs identical to stock

- `onehot_f32` — replaces `feature_utils.bin_and_one_hot` (and its by-name imports in the feature-factory modules): the binned one-hot
  features are built as float32 0/1 directly (`bucketize` + `scatter_`) instead of an int64 one-hot multiplied by 1.0. Numerics: bitwise
  (identical values, no int64 `[…, C]` temporaries). Steps aside: never.
- `target_hoist` — in `ConcatPairFeaturesFactory.forward`, the target × target pair block and the all-zero binder × target
  sequence-separation block depend only on the fixed target: computed at the first network call of a `predict_step` (one dataloader
  batch) and reused by every later step of it. Numerics: bitwise (caching only). Steps aside: a feature configuration other than the
  protein-target one (motif or ligand conditioning, no target coordinates in the batch) takes upstream's own forward on that call,
  counted `fallback_stock`.
- `pair_assembly` — same forward: when every binder in the batch has the same length (a fixed `binder_length [L, L]`; all masks uniform,
  checked once per batch) the extended `[B, N, N, 256]` pair tensor is assembled by four block copies instead of three padded
  concatenations with their host syncs, gathers and mask multiplies. Numerics: bitwise (placement only). Steps aside: binders of differing
  length in the batch (a `binder_length` range) — upstream's padded assembly runs on that call, counted `padded=`; a run whose every call
  was padded reports `state=skipped reason=padded_batches` and `EXIT … gated=pair_assembly:padded_batches`, exit code unaffected. This is
  the kit's only declared gate.
- `loop_desync` — `ProductSpaceFlowMatcher.full_simulation`, `RDNFlowMatcher.simulation_step` and `rdn_flow_matcher.vf_to_score` /
  `score_to_vf`, re-created from their own source with one statement inserted or rewritten each: the sampling loop branches on the
  step's CPU float32 schedule value instead of reading a CUDA scalar back to the host several times per step. Numerics: bitwise (the
  tensors the network sees are unchanged; every comparison stays float32, so each branch decides as stock does). Steps aside: a
  `simulation_step` called outside `full_simulation` takes upstream's own path on that call, counted `fallback_stock`.

Every kit mode also installs two wrappers with no numeric effect: `Proteina.predict_step` (opens the per-batch memo scope the hoisted
blocks and mask checks live in; counts batches) and `LocalLatentsTransformer.forward` (wires the pair-bias source onto the attention
layers at its first call; counts network calls). Their counters feed the `LEVER` / `TALLY` lines and `<out>/kit_records/kit_<pid>.json`.

## fast — within stock's seed-to-seed variation

The default mode: `exact`'s four levers plus the two below.

- `pair_bias_fused` — replaces `PairBiasAttention.forward` as the source of the attention bias: the pair representation is constant
  across the layers (`update_pair_repr` false), so LayerNorm statistics are computed once per network call and all layers' biases come
  from one GEMM with the LayerNorm affine folded into the projection weights (row-chunked; one buffer per shape, refilled in place).
  Numerics: fp32 re-association in the bias. Kernel: PyTorch (`F.layer_norm`, `torch.matmul`); no compiled kernel. Steps aside: a
  transformer whose attention modules are not `PairBiasAttention`, or whose pair representation is updated per layer, keeps upstream's
  path (`not_wired_*` on the LEVER line); a lever that then never engages makes the run exit 3.
- `attn_sdpa` — replaces `PairBiasAttention._attn`: `torch.nn.functional.scaled_dot_product_attention` with the pair bias as the float
  mask (−1e4 at masked pairs, as upstream's own SDPA variant fills; the fill is skipped when the pair mask is all ones) instead of
  einsum → softmax → einsum. Numerics: the SDPA backend's fp32 accumulation order. Kernel: PyTorch SDPA (torch 2.7.0). Steps aside: never.

## big — lowest peak GPU memory

Outputs identical to stock (numerics as `exact`): `exact`'s four levers plus the one below, not `fast`'s two.

- `pair_bias_rows` — replaces `PairBiasAttention.forward` as the source of the attention bias: each layer's `pair_norm → to_bias` runs
  on contiguous row blocks (64 MiB of normalized pair features at a time) written straight into the `[B, h, N, N]` bias, so the
  per-layer `[B, N, N, 256]` LayerNorm output is never materialised. Numerics: bitwise (upstream's own LayerNorm and Linear; each GEMM
  keeps its reduction dimension, only the row count per call changes). Kernel: none added. Steps aside: as `pair_bias_fused`
  (attention class), else never. `pair_bias_rows` and `pair_bias_fused` never install together.

## Every mode

- Stock exceptions: none (STOCK.md §Stock exceptions); no upstream file is patched on any mode, `off` included, and no opt-in upstream
  fixes ship.
- GPU class: no lever depends on the card. `configs/h100.env`, `a100.env` and `h200.env` differ only in `MODEL_OPT_TARGET_GPU`; the
  `ACTIVE` line names the card found by compute capability (`gpu_class=H100|A100|other`; an H200 reads `H100`) and the levers engage
  on any of them.

## Switches

- None that change lever behaviour: the kit reads no lever-ablation or compile switches. `COMPLEXA_OPT=<mode>` names the mode when
  `--mode` is absent (STOCK.md §Variables).
