"""fpf_trimul_v4 — Tier-2 TriangleMultiplication (outgoing+incoming): fused LN+dual-gated-projection prologue writing channel-major bf16 planes (K1),
cuBLAS strided-batched contraction on zero-padded aligned planes (B), fused LN_out + out-projection + gate + residual epilogue (K3).  No permute/transpose/copy kernels.  A leading batch is one launch set (generic: BATCH_MODE).
Entries:
  fpf_trimul_v4.trimul:fn         module-signature provider (STOCK TriangleMultiplication{Outgoing,Incoming}.forward signature; FlashPairformer ARM T plug FPF_SMALLN_TRIMUL_FAST_FN)
  fpf_trimul_v4.generic.trimul    module-agnostic entry (v4.1.0): tensors in, c_z in {128,256}, hidden D in {128,256}, optional biases, bf16 (fp32 accepted) — for any module layout
Credits: stage structure follows the FlashPairformer TriMul v1/v3 kernels (fpf_trimul / fpf_trimul_exact); registry/plug contract = FPF_SPEC_v0 +
the fpf_smalln FAST_FN slot; cuBLAS sm_100 dispatch per the K2B notes."""
__version__ = "4.4.6"
