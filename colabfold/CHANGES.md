# ColabFold kit — what changes vs stock

Stock = ColabFold 1.6.1 / alphafold-colabfold 2.3.13 at the pin (STOCK.md). Each lever is a module the kit installs over one stock class or
function when `colabfold.batch.run` starts under a kit mode (`COLABFOLD_OPT=<mode>`); `off` loads none. A mode is all of its levers; each
mode below includes the previous mode's levers. Lever names are the ones printed on the run's `LEVER name=…` lines; a call a lever does not
take runs the stock operation and is counted there (`fallbacks=<n> fallback_by=<reason>:<n>`). The kernel levers call the shared core's JAX
provider (`opt_core.kernels.pallas.serve`) by the mode's tier word (`fast` | `big`), which names the kernel row per call, or the stock
statement.

## exact — outputs identical to stock

- `DEVICE_RESIDENT` — rebinds `alphafold.model.model.RunModel`: parameters, features and recycling inputs stay on the device across the
  Python recycle loop, outputs are cast to float16 on the device, the loop reads only its per-recycle scalars, and the output tree is
  fetched once after the last recycle (`--save-recycles` / `--save-all`: one fetch per recycle, `fetch=each`). Numerics: bitwise
  (placement only). Steps aside: never.
- `XLA_CACHE` (deployment lever, placed in `exact`, `fast` and `big`; not on `levers=`) — JAX's persistent compilation cache under
  `$COLABFOLD_OPT_JIT_ROOT/<stack key>/<recipe>/jax` via the shared core's `opt_core.capture.xla_cache`; recipe `det` when `XLA_FLAGS` select
  deterministic ops or autotune level 0 (XLA's autotune results then neither loaded nor stored), else `default`; a caller's
  `JAX_COMPILATION_CACHE_DIR` is kept (`source=kept`). Numerics: none. Steps aside: no cache root configured.

## fast — within stock's seed-to-seed variation

This mode binds FlashPairformer kernels via the shared core (opt_core).

- `SUBBATCH` — wraps `alphafold.model.config.model_config`: `global_config.subbatch_size` 128 instead of 4 when the shared core's
  `opt_core.jax_design.subbatch_policy` estimate for the run's largest input fits the visible GPU (`source=auto:fits`), else 4, named
  (`auto:exceeds`, `auto:no_device`). Numerics: same arithmetic in a different XLA program (tolerance class). Steps aside: never.
- `TRIMUL_PALLAS` — rebinds `alphafold.model.modules.TriangleMultiplication` (outgoing / incoming; Evoformer, extra-MSA and template pair
  stacks) to `serve.triangle_multiplication`. Kernel: FlashPairformer triangle multiplication via the shared core — Pallas (prologue and
  epilogue kernels around one batched GEMM; compute capability ≥ 8.0). Numerics: bf16 operands, f32 accumulation and statistics. Steps aside: per call where the
  provider names the stock statement (`stock_by_name`); `n_gpu>1`. Refuses the mode by name below compute capability 8.0 or without the
  Pallas Triton lowering in this jax (`cc_below_8_0`, `no_pallas_triton`, …).
- `AF_PALLAS_ATTN` (`opt/forward/af2_pallas_flash/`) — rebinds `alphafold.model.modules.Attention`: pair-biased calls (triangle attention
  starting / ending node, MSA row attention) make one `serve.attention` call; projections and gating stay stock's. Kernel: FlashPairformer triangle
  attention via the shared core — Pallas (flash attention with the pair bias; bf16 products, f32 softmax statistics and accumulation). Numerics: bf16 re-association, deterministic run to run.
  Steps aside: calls without a pair bias, heads below 16 channels (counted as fallbacks).
- `PALLAS_MSA` — bound over `AF_PALLAS_ATTN`'s class: extra-MSA row attention (8-channel heads) and bias-free MSA column attention through the
  same face. Numerics: as `AF_PALLAS_ATTN`. Steps aside: no provider family for the call (template attention, `no_cell_family`); below a
  provider size rule (`below_keys_rule`, `below_size_rule`).
- `TRIATTN_XLA` — inside `AF_PALLAS_ATTN`'s binding, the pair-biased sites through the shared core's pre-compiled triangle-attention kernels
  (`opt_core.kernels.triattn_xla`; one XLA FFI call per sub-batch chunk, no logits materialised). Kernel: FlashPairformer triangle attention
  via the shared core — CUDA-native (sm_90a / sm_80) and Triton (compiled ahead of time), XLA binding. Numerics: as `AF_PALLAS_ATTN`; a fully masked row attends uniformly, as stock. Steps aside:
  where the provider names another row (`cell`, `below_size_rule`).
- `MSA_COL_CUDNN` — bound over `PALLAS_MSA`'s class: bias-free MSA column attention (8 heads × 32 channels, keys = MSA depth) with every
  provider row admitted. Kernel: cuDNN fused attention in the key-lengths form through jax 0.5.3 on compute capability 9.0; the Pallas row
  elsewhere. Numerics: tolerance class. Steps aside: other calls pass through to the wrapped class.
- `TEMPL_DEDUP` — rebinds `alphafold.model.modules_multimer.TemplateEmbedding`: in the scan over template rows a row equal to its
  predecessor (aatype, positions, mask, compared on the device) reuses the predecessor's embedding — the four identical mock rows of a
  template-free run are embedded once; distinct rows embed as stock (`templates= embedded= reused=`). Numerics: stock's sum over stock's
  operands; rounding-level differences where XLA compiles the branch apart. Steps aside: `monomer_route`, `template_disabled`,
  `dropout_enabled` (`--use-dropout`); `n_gpu>1`.
- `TRANSITION` — binds `alphafold.model.modules.Transition` (Evoformer MSA / pair, extra-MSA, template stack) to `serve.transition`. Kernel:
  FlashPairformer transition via the shared core — Pallas (fused LayerNorm→Linear→ReLU→Linear). Numerics: tolerance class. Steps aside: where the provider names the stock
  statement (`cell_stock`, `no_cell`); `n_gpu>1`.

## big — `fast`'s levers; the pair representation sharded under `--n_gpu`

The levers above, asked by the tier word `big`; at `--n_gpu 1` nothing else is installed and outputs equal `fast`'s.

- `ROWPAIR` (`--n_gpu P`, P ∈ {2, 4, 8}) — AlphaFold-Multimer end to end with the pair representation `[N, N, C]` split by rows over the P
  GPUs of one host through the shared core's `opt_core.mem.rowpair_jax`: pair terms are created row-sharded; the Evoformer, extra-MSA and
  template pair stacks run the stock block bodies on each rank's row block with the shared core's collectives; the recycle carry stays
  sharded; structure-module pair reads and the distogram / aligned-error heads work on rows; the N² outputs are assembled on the host at
  the one fetch. Each rank holds its row block plus the replicated MSA, single representation and parameters. Template pair features are
  formed per rank for its own rows, never as a whole `[T, N, N, F]` (`[colabfold-opt] TEMPLATES job=… real=<r>/<T> form=row_born`). N is
  padded to a multiple of P and cut back before colabfold's writers; a multimer feature outside the kit's residue-axis schema is refused
  by name. `TRIMUL_PALLAS`, `TEMPL_DEDUP` and `TRANSITION` read `state=off reason=n_gpu>1`; `TRIATTN_XLA` and `MSA_COL_CUDNN` serve as on one
  GPU; `SUBBATCH` is 128 (`source=auto:rowpair`). Numerics: as `fast` (complete contractions per element, no cross-device partial sums;
  not bitwise equal to P = 1). Refused by name: P outside {1, 2, 4, 8}, fewer GPUs visible than P, `--n_gpu` > 1 under another mode, a
  shared core without `opt_core.mem.ngpu` / `opt_core.mem.rowpair_jax`. At P = 8 `pred` starts the model process with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` unless the caller exported a fraction.

## Every mode

- The memory-environment stock exception (STOCK.md 'Stock exceptions') applies identically on every mode, `off` included. The kit ships no
  upstream fixes.
- A lever of the mode that cannot run refuses the mode before any compute (`NOT ACTIVE: …; exit 3 (--mode off runs stock)`); a lever no
  call reached in a model-building run is a partial activation (exit 3) unless every call stepped aside under a rule named above or a
  lever bound over it served the calls (`superseded_by=<LEVER>:<n>`); a command line that builds no model prints `IDLE no_model_run=…`.

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers (`ablated=<names>` on the ACTIVE line); `pallas:<row>` /
  `triattn_xla:<row>` switch one provider row off by name. Values and refusals: STOCK.md.
- `AF_PALLAS_ATTN_ALL` and the flash kernel's backward-pass switches `AF_PALLAS_ATTN_PRECISE_BWD`, `AF_PALLAS_ATTN_DBIAS_F32`,
  `AF_PALLAS_ATTN_BWD_BATCH_CHUNK` — registry entries in no mode: the first is refused by name when set; the others change nothing in
  inference.
- `--det 1` — refused by name; no deterministic recipe is set by the kit (README Notes).
