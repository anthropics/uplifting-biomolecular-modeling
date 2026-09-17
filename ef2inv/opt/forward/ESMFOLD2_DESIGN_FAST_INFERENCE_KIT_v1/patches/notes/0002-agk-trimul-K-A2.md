# K-A2 — triangle multiplication fwd+bwd under autograd (`agk.enable(model, trimul='bmm2')`)

**Applies to:** the pinned Biohub transformers fork (`models/esmfold2`) and the esm cookbook's `binder_design` (STOCK.md §Pin). Applied at run
time by patching bound methods of the loaded model; no source edits.

## What
Stock ESMFold2 runs the reference TriangleMultiplication path whenever grad is enabled (the vendored Triton kernels and the cuEquivariance
backend disable themselves or cover only part of it). K-A2 wires the vendored Triton LN / gated-dual-GEMM / gated-GEMM-with-residual kernels
under autograd with a frozen-weight backward (no dW work) and replaces the chunked fp32 einsum by one cuBLAS batched GEMM on the native
(d, b, i, k) layout (no permute copies; bf16 operands, fp32 accumulation, bf16 out). No mode selects it: `ef2_trimul` variant `fused` is the
triangle multiplication of fast / big.

## Numerics class
fast: rounding points changed — contraction operands bf16 instead of fp32-of-bf16 values, gate / projection epilogues fused; not bitwise to stock.
