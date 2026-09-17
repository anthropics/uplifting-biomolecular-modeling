# protenix_fpf_dit_attn_exact — attribution

- `csrc/dit_attn_exact.cu`: original CUDA source. The order of floating-point operations it reproduces (3xTF32 operand split and mma pass
  order, staged accumulation grouping, softmax formulation with exp2, online rescaling, row-sum reduction order, reciprocal epilogue) is that of
  the memory-efficient attention forward kernel of PyTorch (`aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`,
  itself derived from xFormers), and of the CUTLASS components it instantiates (`cutlass/gemm/warp/mma_tensor_op_fast_f32.h`,
  `cutlass/numeric_conversion.h`). No source text was copied from those files.
- PyTorch: BSD-3-Clause licence, Copyright (c) 2016- Facebook, Inc (Adam Paszke) and other contributors. xFormers: BSD-3-Clause,
  Copyright (c) Facebook, Inc. and its affiliates. CUTLASS: BSD-3-Clause, Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES.
- `prebuilt/*/dit_attn_exact.so`: compiled from `csrc/dit_attn_exact.cu` with nvcc against the installed torch headers (see PROVENANCE.md).
