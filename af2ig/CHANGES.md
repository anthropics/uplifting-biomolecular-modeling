# AF2 initial guess (dl_binder_design) kit — what changes vs stock

Stock = dl_binder_design `af2_initial_guess` @ `cafa3853` at the pin (STOCK.md), run through `predict_pdb.py`. Each lever is code
the kit's patch series adds to the pinned tree behind an opt-in flag of `predict_pdb.py` (or a variable of its process), engaged
only under a kit mode; `off` passes none. A mode is all of its levers; each mode below includes the previous mode's levers unless a
line says otherwise. Lever names are the ones printed on the run's `LEVER name=…` lines.

## exact — outputs identical to stock

Bitwise with `off` under `--det 1`.

- `L6` (`-host_outputs`) — the model's output tree copied to the host in one `jax.device_get` before post-processing. Numerics:
  bitwise (placement only). Steps aside: never.
- `L1` (`-device_params`) — the parameter tree placed on the GPU once at start-up instead of a host copy per design. Numerics:
  bitwise. Steps aside: never.
- `U1` (`-sort_by_length`) — inputs processed in (residue count, name) order; `L6 L1 U1` travel as the driver's `-fast` preset.
  Numerics: bitwise. Steps aside: never.
- `L7` (`-precompile N`) — the programs of the run's distinct residue counts compiled through jax's ahead-of-time path on N host
  threads (default 6): the loop's first length alone first, the rest in the background while designs run; no forward runs in the
  pass (`opt/af2ig_opt/precompile.py`). Numerics: bitwise (compilation scheduling only). Steps aside: never.
- `L13` (`-program_cache DIR`) — each compiled program serialized under `<jit root>/<stack key>/<recipe>/programs` and loaded by
  later processes instead of traced and compiled; the key covers the code of the checkout, this package and the shared core, the
  lever arguments, stack, device and XLA flags, so any change is a cold program by name; a file that does not load is named
  (`load_failed:<Type>`) and traced (`opt/af2ig_opt/programs.py`). Numerics: bitwise. Steps aside: together with `ccache`.
- `ccache` (`JAX_COMPILATION_CACHE_DIR=<jit root>/<stack key>/<recipe>/jax` in the driver's environment, through the shared core's
  `opt_core.capture.xla_cache`) — JAX's persistent compilation cache, every compilation stored; `<recipe>` keeps `--det` runs
  apart from default-numerics runs (under `--det` no autotune result is read from disk: `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`).
  Numerics: bitwise. Steps aside: `reason=cache_dir_unwritable:<dir>` when the root cannot be written (`skipped=ccache` on the ACTIVE line; the program store then goes to the default root, and `L13` steps aside the same way only if that root cannot be written either).
- `L15` (`-prefetch 2`) — the next two designs read and featurised on one worker thread while the current design runs; same code,
  same order (`opt/af2ig_opt/prefetch.py`). Numerics: bitwise. Steps aside: never.
- `L16` (`-overlap_output 1`) — each design's output step (confidence metrics, RMSDs, PDB, score line, checkpoint line) on one
  writer thread behind the loop, in the loop's order. Numerics: bitwise. Steps aside: never.

## fast — within stock's seed-to-seed variation

Kit-true class: not bitwise — TF32 products and bf16 triangle-attention operands, float32 elsewhere (stock has no seed axis; the
reference is stock's own output). This mode binds FlashPairformer kernels via the shared core (opt_core).

- `L9` (`-subbatch 128`) — the inference sub-batch of every Attention module and Transition raised from stock's 4 rows to 128
  (`opt/af2ig_opt/subbatch.py` over the shared core's `opt_core.jax_design.subbatch_policy`). Numerics: the same arithmetic per
  row; XLA may pick other GEMM algorithms for the larger shapes. Steps aside: never.
- `L8` (`-flash_attn`) — the attention core of the pair-biased `Attention` calls (MSA row attention; the template pair stack's
  attention when `L10` is off) computed by the shared core's flash attention (`opt_core.kernels.pallas.serve.attention`: Pallas,
  tile tables for sm_90 and sm_80); MSA-column and template point-wise attention stay on stock ops (`opt/af2ig_opt/flash_attn.py`).
  Numerics: TF32 products, softmax and accumulation in float32 inside the kernel. Steps aside: a call whose operands are not
  float32 stays on the stock body (`reason=dtype`).
- `L10` (`-fused_triattn`) — every `TriangleAttention` module (starting / ending node; Evoformer and template pair stacks) served
  as one fused block by `opt_core.kernels.pallas.serve.triangle_attention_block` (`opt/af2ig_opt/triattn.py`). Kernel: FlashPairformer
  triangle attention via the shared core — the provider's row per call (Pallas, or the XLA binding). Numerics: as `L8`. Steps aside: `reason=cannot_run:…` on a GPU the shared core has no float32 tile table for (together with `L11`); per call as `L8`.
- `L19` (`AF2IG_OPT_TRIATTN_CORE_DTYPE=bf16`, placed in the driver's environment) — the TriangleAttention module handed to the
  provider in bfloat16: activations and parameters cast at the module boundary, pair bias, softmax and accumulation in float32,
  output back to float32; `=fp32` keeps float32 operands. Numerics: bf16 operand rounding. Steps aside:
  `reason=cannot_run:needs_L10` without `L10`.
- `L11` (`-fused_trimul`) — every `TriangleMultiplication` module (outgoing / incoming) served as one fused block by
  `opt_core.kernels.pallas.serve.triangle_multiplication` (`opt/af2ig_opt/fused_trimul.py`). Kernel: FlashPairformer triangle
  multiplication via the shared core — Pallas. Numerics: TF32 products. Steps aside: as `L10`; under `big` with `trimul_chunk` opted in, `state=off reason=superseded_by:trimul_chunk` at chunked lengths.
- `L12` (`-opm_reassoc`) — `OuterProductMean` re-associated: the output weight folded into the right projection, then one GEMM
  over (MSA row, channel), without the `[rows, N, 32×32]` intermediate; plain JAX (`opt/af2ig_opt/opm.py`). Numerics: fp32
  re-association. Steps aside: never.
- `L18` (`-tmpl_pointwise_sub 8192`) — the template point-wise attention sub-batch (`template.subbatch_size`, stock 128 rows) at
  8192 rows; `AF2IG_OPT_TMPL_POINTWISE_SUB=<rows>|off` overrides it. Numerics: as `L9`. Steps aside: never.

The kernel levers (`L8`, `L10`, `L11`) pass the provider the mode's tier word and the product class `tf32`; the provider picks each
call's implementation for the GPU's compute capability, dtype and shape and reports it on the LEVER lines (`providers=…`,
`fallback=<n>`); no served call, an undeclared fallback or a kernel error makes the run partial (README.md Notes).

## big — lowest peak GPU memory

Numerics as `fast`. `fast`'s levers without `L18` (the template attention keeps stock's 128 rows), on one GPU (the kit has no
multi-GPU line), plus:

- `mem_fraction` (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` in the driver's environment; `opt/af2ig_opt/big.py`) — XLA's memory pool
  at 95 % of the card instead of the default 75 %. Numerics: none. Steps aside: never.
- `trimul_chunk` (`-trimul_chunk <rows>:<min residues>`; opt-in only: `AF2IG_OPT_TRIMUL_CHUNK=on` = `256:1473`, or
  `<rows>:<min residues>`) — `TriangleMultiplication` evaluated in row chunks of the pair representation from the floor length up
  (`opt/af2ig_opt/pairstack.py` over the shared core's `opt_core.mem.rowpair_jax.rowchunk`); at chunked lengths it takes every call
  and `L11` reads `state=off reason=superseded_by:trimul_chunk`. Numerics: stock's fp32 arithmetic per chunk. No shipped mode composes it.

## Every mode

- The PDB front end and import shims (STOCK.md §Stock exceptions) apply identically on every mode, `off` included. `--det 1` = the
  shared core's XLA recipe (`opt_core.precision.xla`: `XLA_FLAGS=--xla_gpu_autotune_level=0`) on every mode, `off` included.
- Refused by name before any design (exit 3): a checkout that is not the pinned patched tree, an absent pinned distribution or
  shared-core module, a missing parameter file (`AF2IG_OPT_FORCE=1` overrides). Named and run: version drift (`note='pins drift …'`),
  a GPU other than `MODEL_OPT_TARGET_GPU`.

## Switches

- `MODEL_OPT_LEVERS_OFF=<lever>[,…]` — a mode without the named levers (`levers_off=<ids>` on the ACTIVE line; ids that are not
  levers of the mode are named and ignored). `--no-compile` = `MODEL_OPT_LEVERS_OFF=compile`, which drops nothing here
  (`compile=stock_jit`: the kit adds no compile step).
- `--det 1` — above. `--precompile N` — `L7`'s thread count. `--seed N` — model seed N through a seeded copy of the driver built
  for the run (`0` = the stock, unseeded driver). `--allow-partial` — README.md Notes. Kit variables that change lever behaviour:
  `AF2IG_OPT_TRIMUL_CHUNK`, `AF2IG_OPT_TMPL_POINTWISE_SUB`, `AF2IG_OPT_TRIATTN_CORE_DTYPE` (values and defaults: STOCK.md).
