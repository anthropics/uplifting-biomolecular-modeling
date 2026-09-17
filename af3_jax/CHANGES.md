# AlphaFold 3 (JAX) kit — what changes vs stock

Stock = AlphaFold 3 (JAX) `v3.1.4` at the pin (STOCK.md). `off` runs the fork's `run_alphafold.py`; every kit mode runs the kit's copy
`run_alphafold_fast.py` (= `run_alphafold.py` + `opt/forward/fast_inference/patches/01_run_alphafold_fast_inference.diff`) under the stock
interpreter through a launcher (`opt/af3_jax_opt/levers_launch.py` | `fpf_launch.py` | `big_launch.py`) that rebinds the named Haiku modules of
`alphafold3.model` in that one process before the first forward; `off` loads none. A mode is all of its levers (`modes.py` is the table,
`registry.py` the descriptions); lever names are the ones on the run's `LEVER name=…` lines. Kernels come from the shared core through its provider
`opt_core.kernels.pallas` by the mode's tier word (`exact` | `fast` | `big`). A call a kernel does not serve keeps that call site's own arithmetic,
counted by name on the launcher's `SERVED …` line; such a lever, or a cold cache class, is named on the `DONE` line's `partial=` token with
`status=ok` (exit 0). A lever of the mode that engaged nowhere makes the pass `DONE status=partial`, exit 3 (`cli.py`: a mode is all of its levers).

## exact — outputs identical to stock

Outputs identical to `off` inside one compile-cache class on one card (README Notes).

- `L1` — upcoming (input, seed) items are featurised in 3 worker processes with the identical `featurise_input` call while the GPU runs the current
  item (`--featurisation_workers=3 --featurisation_prefetch=4`). Numerics: bitwise (scheduling only). Steps aside: never.
- `WRITER` — each fold job's result extraction and `write_outputs` run on one writer thread, in submission order, behind the next job's inference
  (`--output_writer`). Numerics: bitwise (same calls, later). Steps aside: never.
- `FIX1` — stock calls `tokamax.autotune()` before the first inference of every new bucket size and the call raises on this stack; the kit script
  makes the attempt once per process and prints the cause once (`TOKAMAX …` line). Numerics: none. Steps aside: never.
- `GLUT` — `TriangleMultiplication`'s gated linear unit → transpose → mask as one kernel whose epilogue stores each tile transposed and masked;
  stock's dtype chain and rounding points. Numerics: bitwise. Kernel: shared core `pallas_glut` (Pallas-Triton) via `serve.glu_transposed_masked`.
- `ATTNCFG` — pins tokamax's flash-attention tile configuration for the pair attention (`GridSelfAttention`) to `64,64,4,3`, inside the bitwise
  class of configurations. Numerics: bitwise. Steps aside: never.

## fast — within stock's seed-to-seed variation

This mode binds FlashPairformer kernels via the shared core (opt_core). `L1`, `WRITER` and `FIX1` carry over from `exact`; `GLUT` and
`ATTNCFG` are `exact`'s only; the levers below are added (`fast` is the default mode).

- `FPF_TRIMUL` — `TriangleMultiplication` → prologue kernel (input LayerNorm, projection + gate + mask as channel-major bf16 planes), the cubic
  contraction on cuBLAS, epilogue kernel (centre LayerNorm, output projection, gating). Numerics: bf16 re-association. Kernel: FlashPairformer
  triangle multiplication via the shared core — Pallas. Steps aside: a call outside the served shapes runs the stock class (`fallback=`).
- `FPF_TRIATT` — `GridSelfAttention` → prologue, flash-attention forward for `[N, N, 4, 32]` with pair bias and key mask (online softmax, f32
  accumulation), epilogue; the ending node reads the pair activation transposed through the BlockSpec. Numerics: bf16 re-association. Kernel: FlashPairformer
  triangle attention via the shared core — Pallas. Steps aside: as `FPF_TRIMUL`.
- `TRIATT_XLA` — the triangle attention's core (trunk, MSA, template and confidence-head pairformers) decided by `serve.attention`: where the
  provider names it, pre-compiled kernels through one XLA-FFI target. Kernel: FlashPairformer triangle attention via the shared core —
  CUDA-native (sm_90a) | Triton (ahead-of-time compiled) | XLA binding. Numerics: bf16 products, f32 online softmax. Steps aside: an unserved dtype, head size or card keeps that site's kernel (`txla_aside=`).
- `TRIMUL_CD` — the triangle multiplication served whole through `serve.triangle_multiplication` (prologue kernel → one batched cuBLAS GEMM on
  channel-major planes → epilogue kernel). Kernel: FlashPairformer triangle multiplication via the shared core — Pallas | XLA binding. Numerics:
  fused-kernel rounding. Steps aside: a non-bf16, non-square or unmasked call (`tcd_aside=`).
- `TTR` — the pair transition (`TransitionBlock`): LayerNorm + SwiGLU + output projection in one kernel through `serve.transition`; the
  `[N, N, 4C]` intermediate never reaches HBM. Kernel: FlashPairformer transition via the shared core — Pallas. Numerics: bf16 re-association.
  Steps aside: the single (c 384) transitions stay stock by name (`ttr_routed=`).
- `DATTN` — the pairformer's single attention and the diffusion transformer's self-attention (dense, pair-biased) through `serve.attention`
  (tokamax 0.0.12 flash attention). Numerics: flash-attention accumulation order. Steps aside: per the provider's table (`DATTN word=… served=…` line).
- `LNP` — standalone LayerNorms of the pair, template-pair and MSA units through `serve.layer_norm` (shared core row-LayerNorm, Pallas, where its
  table names it; XLA elsewhere). Numerics: f32 statistics, another summation order. Steps aside: adaptive / single / diffusion norms by rule; `n_gpu>1`.
- `FPF_HOIST` — the diffusion head's step-invariant pair conditioning and the 24 per-block pair-logit projections computed once per sample call
  instead of at every one of the 200 denoising steps; holds 24 × `[16, N, N]` pair logits for the call. Numerics: bf16 re-association. Steps aside: never under `fast`.
- `HOIST_LOGITS` — the diffusion-transformer blocks index the hoisted `[24, 16, N, N]` logits in place (`dynamic_index_in_dim`) instead of
  re-slicing them per block per step. Numerics: identical to `fast` without it. Steps aside: without `FPF_HOIST` (`hlog=skipped`).
- `COND_SHARE` — the sampler's per-step conditioning that depends on the noise level alone is traced once per step and shared by that step's
  samples; replaces `diffusion_head.sample`. Numerics: other GEMM shapes, ulp-level. Steps aside: `n_gpu>1`.
- `ATOM_COND_HOIST` — the atom cross-attention encoder's step-invariant conditioning and both atom transformers' per-block pair logits computed
  once per sample call and handed to the rebound encoder / decoder. Numerics: identical to `fast` without it. Steps aside: without `FPF_HOIST` (`achoist=skipped`).
- `SAMPLER_BF16` — the diffusion sampler's transformer blocks on bf16 tensor-core operands (`adaptive_layernorm` / `adaptive_zero_init` rebound
  inside `diffusion_head.sample` only; residual streams, statistics, `precision='highest'` projections and conditioning stay f32). Steps aside: never.
- `ATOM_ATTN` — the windowed atom cross-attention (query subsets of 32 atoms against key windows of 128) as one Pallas kernel per call
  (`inprocess/af3_jax_atom_attn_pallas.py`: q·k in f32 with bias and mask folded in, softmax and ·v in the same program). Numerics: summation
  order. Steps aside: a shape it does not tile keeps the stock lines (`atomattn_fallback=`).

Install order in the model process is `fpf_launch.TREE_LEVERS`: `COND_SHARE` first (it replaces `sample`), the other in-process levers before the
FlashPairformer import (which derives its fused classes from the rebound modules), `HOIST_LOGITS` after it.

## big — lowest peak GPU memory

Numerics as `fast`. Decided before launch from the input's token estimate (`--n_est` over it). At or below N* = 1,408 padded tokens on one GPU `big` runs `fast`'s
program and cache class. Above it: `fast`'s levers with the tier word `big`, minus `FPF_HOIST` (with `HOIST_LOGITS`, `ATOM_COND_HOIST`),
`FPF_TRIMUL` and `TRIMUL_CD`, off by property (`GLUT` serves the triangle multiplication); `FPF_TRIATT` + `TTR` stay up to their serve edge of
5,120 padded tokens on compute capability ≥ 9.0 and are off by name below it or past the edge (`ATTNCFG` then takes the attention site); plus the
memory levers below. The `BIG NOTE region=… n_est=… n_star=1408` line names what the region leaves inactive and why.

- `TRANSITION_SHARD` — the pair transitions in row shards of 256 through upstream's `pair_transition_shard_spec`. Steps aside: never on the memory line.
- `COND_SHARD` — the diffusion head's pair conditioning in row shards of 256 through the fork's `mapping.sharded_apply`. Steps aside: `n_gpu>1`.
- `TRIMUL_CHUNK` — the trunk's triangle multiplication in row blocks of 512 of the output, composed on `GLUT`. Steps aside: `n_gpu>1`.
- `SAMPLES_PER_PASS` — one diffusion sample per pass with stock's random draws kept; registered, composed on no line.
- `LOGITS_SHARD` — the diffusion transformer's per-block pair logits in row shards; registered, composed on no line.
### `--n_gpu P`

- `ROWPAIR` (P = 2, 4, 8; one process on one host) — each device holds a row block `[N/P, N, C]` of the pair representation from the
  embedder on; the trunk pair stack with its recycling carry, the template embedder (template pair features formed per row block, `TEMPLATES …
  form=row_born`), the MSA pair reads, the distogram and confidence heads and the diffusion pair conditioning run per device on its own rows,
  collectives completing every contraction (triangle multiplication on a ring schedule, `schedule=ring`); single and atom-level tensors are
  replicated (`inprocess/rowpair.py` onto the shared core's row-sharded pair stack, attached at the kit script's `_jitted_apply`). Supersedes
  `COND_SHARD`, `LOGITS_SHARD`, `TRIMUL_CHUNK`, `GLUT`, `DATTN` and `LNP` (`superseded=` on the ACTIVE line); `FPF_TRIATT`, `TTR` and `COND_SHARE`
  are off on this line. Refused by name: `--n_gpu` > 1 under another mode, fewer than P GPUs visible, a memory-pool fraction above the line's
  ceiling (at `--n_gpu 8` the kit sets 0.9, `NOTE xla_mem_fraction=…`).

## Every mode

- The stock exception (STOCK.md §Stock exceptions) applies identically on every mode, `off` included.
- The kit modes key their compile-cache class on the GPU model name instead of jax's device-topology fingerprint (`inprocess/portable_cache_key.py`,
  `CACHE … cache_key=device_kind`; `off` keeps upstream's key). The `KERNELS … verdict=…` gate holds each attention site to the implementation its
  lever declares (absent or fallen back: exit 5). Every stock flag passes through; a caller's `--cache_dir` on a kit mode is the class its levers use.

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers (`ablated=<names>` on the ACTIVE line); `triattn_xla:<row>` switches one
  kernel-provider row off by name (`rows_off=`). `FIX1`, `ROWPAIR` and any name under `off` are refused by name. `AF3_JAX_TRIMUL_CD=<word>` in the
  model process names another arm of the triangle-multiplication op for A/B work; every other lever switch is written by the mode table (STOCK.md).
