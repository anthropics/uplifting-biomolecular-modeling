# Boltz-2 kit — what changes vs stock

Stock = `boltz` 2.2.1 at the pin (STOCK.md). Each lever is a module the kit installs over one stock function or class when its
persistent worker builds the model under a kit mode; `off` loads none. Each mode below includes the previous mode's levers unless a line
says otherwise; lever names are the ones on the run's `LEVER name=…` lines, and every way a lever leaves a mode is printed by name.

## exact — outputs identical to stock
This mode binds FlashPairformer kernels via the shared core (opt_core).
- `resid`, `mask2` — Pairformer eval-mode shortcuts: the all-ones dropout mask and the all-zero mask term are not multiplied in. Bitwise.
- `fpf_trimul_exact` — triangle multiplication through the core's provider (tier word `exact`): the row its table names for the card
  and size, else cuEquivariance's op by upstream's own call (`fallback_by=library_op:cueq`). Numerics: bitwise. Kernel: FlashPairformer
  triangle multiplication via the shared core — CUDA-native (sm_90a / sm_80) or Triton, as the table names; exact variant.
- `fused_transition` — the pair Transition (LayerNorm → W_a|W_b → silu(a)·b → W_out) as one kernel where the table holds a bitwise row,
  else the module's own forward, by name. Numerics: bitwise. Kernel: FlashPairformer transition via the shared core — Triton; exact variant.
- `pairblock` — the triangle-attention block as prologue kernel + attention core + epilogue kernel; the attention core is
  cuEquivariance's triangle attention, the call stock makes (no kit kernel for the attention itself). Numerics: bitwise. Kernel:
  the shared core's fused pair-block prologue / epilogue — Triton; exact variant. Steps aside: below 128 tokens, C ≠ 128; on compute capability
  8.0 outside the row range on the LEVER line (`min_rows=` / `max_rows=` / `piece_rows=`).
- `triattn_exact` — `pairblock`'s attention core asked of the shared core's triangle-attention provider by its tier word `exact`: the
  core's exact-class kernel row on a (card, torch, library) stack its table vouches for, cuEquivariance's call — the one stock makes —
  everywhere else; the row for this card and stack is on the LEVER line (`row=`, `stack=`). Numerics: bitwise.
- `exactln`, `exactln_resid` — a bitwise replica of ATen's vectorized LayerNorm for the [N,N,128] pair LayerNorms (the shared core's
  LayerNorm kernel, Triton), the pair residual add fused into the next LayerNorm pass; a shape class it cannot prove steps aside by name.
- `templ_skip`, `graph_trunk` — an all-dummy template pass skips the template module (its update is +0.0); the Pairformer stacks are
  CUDA-graph captured and replayed per (module, shape) up to `BOLTZ_GRAPH_TRUNK_MAX_TOKENS` (300) tokens. Bitwise.
- `rollout`, `dit_hoist`, `align_aligncap` — the diffusion sampler as a captured roll-out, the step-invariant token-transformer work
  computed once per sample, the per-step rigid-alignment SVD on stock's cuSOLVER call between replays. Numerics: bitwise. Steps aside:
  `--use_potentials`, unequal `--max_parallel_samples` chunks (`distinct_sample_chunks`), and per prediction when the projected working
  set does not fit free GPU memory (`headroom_gated=<n>`) — stock's eager loop then.
- `dit_par`, `dit_mask`, `dit_sba`, `dit_smx`, `dit_glue` — the token transformer's 24 layers re-scheduled inside the captured step: the
  all-zero mask term skipped; scale + pair bias + softmax and the elementwise glue as NVRTC-compiled CUDA kernels keeping ATen's op order
  (`opt/forward/ditexact`). Numerics: bitwise, proven per shape class at first use. Steps aside: with `dit_hoist`.
- `waste_chunkcast`, `waste_opmmask`, `waste_opmdiv`, `waste_ctorskip` (switch words `chunkcast`, `opmmask`, `opmdiv`, `ctorskip`) — the
  MSA module's chunked paths cast to bf16 once, reuse the mask count, divide in place; the constructor skips the initialisation the
  checkpoint load overwrites. Numerics: bitwise.
- `msa_pwa_exact`, `msa_trans2_exact` — pair-weighted averaging and the dim-64 transitions as fused Triton kernels in stock-rounding mode
  (`opt_core.ops.msa_pwa2`, `opt/forward/msa2/trans2.py`). Numerics: bitwise. Steps aside: `msa_pwa_exact` on compute capability 8.0.
- `atom_keys_gather`, `atom_glue_hoist` — the atom windowing einsum as a gather kernel; the atom encoder/decoder's step-invariant
  statements hoisted to each sample's first step. Bitwise. `writer_overlap`, `prefetch` — boltz's writer on a background process (same
  bytes, joined before EXIT); the next input featurized by a persistent helper meanwhile — CPU storages of 4 GiB and up in that hand-over
  travel as files on local disk (`BOLTZ_OPT_XFER`), every wait has a deadline that fails by name; census on the `[boltz2-opt] XFER …` line.
## fast — within stock's seed-to-seed variation
`exact`'s levers except `exactln_resid`, `dit_par…dit_glue`, `msa_pwa_exact`, `triattn_exact`; `fpf_trimul_exact` → `fpf_trimul`,
`msa_trans2_exact` → `msa_trans2`, `align_aligncap` → `align_jacobi64`, `atom_glue_hoist` → `atom_fused`; plus:
- `pairfuse` — every C=128 pair stack (trunk, MSA-module pair layers, confidence) driven on one resident bf16 pair tensor, each sub-layer
  reading and writing z once; `pairblock`, `fused_transition`, `fpf_trimul` serve inside it by the providers' `fast` tier words and the
  served row is printed (`CORE-CELL site=… row=<row>`). Numerics: bf16 residual stream, fused-kernel rounding. Kernel: FlashPairformer
  triangle attention | triangle multiplication | transition via the shared core — CUDA-native (sm_90a / sm_80) or Triton; fast variant.
- `flash_triattn`, `pairblock_c64` — fused online-softmax triangle attention from `BOLTZ_TRIATTN_MIN_TOKENS` (300) tokens (cuEquivariance's
  below), also for the template Pairformer's C=64 stack. Numerics: bf16 re-association. Kernel: as `pairfuse`, triangle attention.
- `fpf_opm`, `fpf_pwa`, `msa_trans2` — outer-product mean, pair-weighted averaging and dim-64 transitions as fused Triton kernels
  (`opt_core.ops.msa_opm` / `msa_pwa`, `opt/forward/msa2`). Numerics: fused-kernel rounding. Steps aside: `fpf_opm` below 385 tokens on 8.0.
- `dit_fused`, `align_jacobi64` — the diffusion token-transformer step as one fused bf16 Triton schedule inside the roll-out; the rigid
  alignment's 3×3 SVD + determinant as one in-graph fp64 Jacobi kernel (`opt/forward/sampler/src`). Numerics: bf16 operands, fp32 accumulation.
- `condproj`; `atom_fused`, `atom_gemm` — DiffusionConditioning's 24 + 3 + 3 pair-bias projections as one fused pass; the atom
  encoder + decoder as 8 + 8 Triton launches per step (`opt/forward/atom/src`). Numerics: bf16 operands, fp32 accumulation.
## big — lowest peak GPU memory
`fast`'s levers, numerics as `fast`, with these off by the row's own entries (printed `state=off` with the reason): `rollout`,
`graph_trunk` (`rule:big_no_graphs`), `dit_hoist`, `align_jacobi64` (`needs:rollout`), `fpf_pwa` (`measured_memory_2.8GiB_alloc_peak`),
`condproj` (`site_taken_by:xl_cond`), `atom_keys_gather` where `atom_fused` serves; `dit_fused` serves every diffusion step from stock's
eager loop with per-sample weight packing; `atom_fused` / `atom_gemm` off on compute capability 8.0 (`card_off`); plus:
- `dit_tf32` — TF32 tensor-core products for the diffusion module's remaining fp32 GEMMs. `xl_trans`, `xl_cond` — the pair Transition
  and the diffusion pair conditioner evaluated 256 rows at a time into one preallocated buffer (`opt_core.mem.chunk`; `xl_cond` in fp32).
  `xl_free`, `relpos_lazy`, `expandable_segments` — dead conditioning outputs freed at once; the fp32 relative-position encoding released
  after its one consumer and recomputed where read again; the expandable-segments allocator.
### `--n_gpu P`
P ∈ {2, 4, 8}, `big` only (lever `rowpair_tp`): the trunk, MSA-module, template and confidence pair stacks run with the pair
representation row-sharded over P GPUs of one host (one worker rank per GPU, NCCL; `opt_core.mem.rowpair` statements with the core's
triangle kernels per row window; the (z + zᵀ) readers stream row blocks sized by the widest shard, so every rank walks the same
all-to-all rounds when N is not a multiple of P); each rank runs the diffusion sampler as stock's eager loop. Rank 0 parses, featurizes and broadcasts
each batch (`data_form=rank0_bcast`, `feats_ranks_equal=yes`), its outputs are the run's, an affinity pass runs on rank 0 only, and
templated inputs build their template pair rows per rank (`form=row_born`). `tp_off=` at P > 1: `graph_trunk`, `pairblock`,
`pairblock_c64`, `fused_transition`, `pairfuse`, `templ_skip`, `rollout`, `dit_hoist`, `align_jacobi64`, `dit_fused`, `atom_fused`,
`atom_gemm`, `writer_overlap`. Refused by name: P not in {2, 4, 8}, fewer GPUs visible, any mode but `big`. All ranks share one
`PYTHONHASHSEED` (`RANKENV hashseed=<v>`); the launcher's placement words (`ROWPAIR_*`) are printed on the LEVER line.

## Every mode
- No stock exception: every mode runs `boltz` 2.2.1 as shipped, kernels on; `--no_kernels` and `--det` belong to `off`.
- The `templ` guard prints `TEMPLATES record=<id> … real=<live>/<slots>` per templated input and changes no output. Compute capability
  below 8.0 is refused by name; on 10.0 the levers without a table for the card leave by name (`card_off=`).
- An (input, seed) that runs out of GPU memory is reported `FAILED item=<name> seed=<s> reason=…`; the run continues (exit 1).
## Switches
- `--no-compile` (= `MODEL_OPT_LEVERS_OFF=compile`) — accepted on every verb; no mode uses `torch.compile` (`compile=off:none_in_kit`
  on the ACTIVE line, `off:user` with the flag).
- `BOLTZ_OPT_XFER=auto|shm|file` (default `auto`; `BOLTZ_OPT_XFER_MIN_MIB` 4096, `BOLTZ_OPT_XFER_DIR`) — the featurizer hand-over's transport,
  files at and above the floor; `BOLTZ_OPT_HELPER_TIMEOUT_S` (3600) / `BOLTZ_OPT_LOADER_TIMEOUT_S` (1800) — its deadlines, failing by name.
- The `BOLTZ_*` switch words in `opt/boltz2_opt/registry.py` compose a mode inside the worker; an undeclared `BOLTZ2_OPT*` name is
  refused. Levers no mode sets: `graph_sampler`, `dit_bf16`, `dit_attn_bf16`, `seq_bf16`, `tfeat`, `tdummy`, `msa_hoist`, `mask`, `cast`, `tln`, `qkvg`.
