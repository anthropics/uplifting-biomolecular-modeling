# af2_pallas_flash — flash attention for AlphaFold-2 / AF-Multimer (`alphafold-colabfold`) in JAX Pallas

**What it is.** `af2_pallas_flash/af2_pallas_attn.py` rebinds `alphafold.model.modules.Attention` at run time so that the attention core —
softmax(q·kᵀ·scale + mask bias + pair bias)·v — runs through the Pallas (Triton backend) flash-attention kernel of the shared core
(`opt_core/kernels/pallas_attn/af2_flash_pallas.py`, imported here as `af2_flash_pallas`) instead of materialising the `[B, H, Q, K]` logits.
The query / key / value projections, the gating and the output projection are the stock einsums on the stock parameters, and the replacement
class keeps the name `Attention`, so the haiku parameter tree is unchanged and stock checkpoints apply as they are. No upstream file is modified.

**Which calls it takes.** Those carrying a non-batched pair bias — triangle attention (starting / ending node) and MSA row attention with pair
bias; with `AF_PALLAS_ATTN_ALL=1` also the calls without one (MSA column and template attention). A call the kernel cannot take — per-head
dimension below 16 or key_dim ≠ value_dim (template-pair attention has 8 per head), a bias not of the AlphaFold-2 form, a non-GPU backend — runs
the stock arithmetic and is counted in `_STATE['fallbacks']`; served calls are counted in `_STATE['calls']` (`af2_pallas_attn.py:19-30,58-66`).

**Numerics.** Not bit-identical to stock XLA: the bf16 attention arithmetic is re-associated (flash-attention order, fp32 accumulation, exact
online softmax), so outputs differ from stock at the bf16 rounding level; the model's precision policy is unchanged (`global_config.bfloat16`).
The kernel is deterministic run to run: the same inputs give the same bytes.

**Requirements.**
- an NVIDIA GPU of compute capability ≥ 8.0;
- `jax[cuda12]` 0.5.3 – 0.7.x (the Pallas Triton lowering ships inside jaxlib: no separate `triton` package and no C compiler are needed);
- `dm-haiku` and `alphafold-colabfold` 2.3.x (the `Attention` class it rebinds), in the process that builds the model.

**Use.** Put `af2_pallas_flash/` on `sys.path`, make the module name `af2_flash_pallas` importable (the shared core's kernel), then either
`export AF_PALLAS_ATTN=1` and `import af2_pallas_attn` (the module calls `enable()` at import when the switch is set, `af2_pallas_attn.py:124-125`)
or call `af2_pallas_attn.enable()` yourself — in both cases BEFORE the model is built or jitted; `disable()` restores the stock class.
Switches read: `AF_PALLAS_ATTN` (enable at import) and `AF_PALLAS_ATTN_ALL` (all attention calls; default `0`). The kernel's backward-pass
switches (`AF_PALLAS_ATTN_PRECISE_BWD`, `AF_PALLAS_ATTN_DBIAS_F32`, `AF_PALLAS_ATTN_BWD_BATCH_CHUNK`) change nothing in a forward-only run.

**Contents.** `af2_pallas_flash/af2_pallas_attn.py`, `NOTICE` (third-party attributions).
