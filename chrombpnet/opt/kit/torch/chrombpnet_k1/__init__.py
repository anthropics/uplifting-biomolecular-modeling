"""chrombpnet_k1 — the ChromBPNet forward as Triton kernels on the torch route (weights loaded through bpnet-lite's ChromBPNet.from_chrombpnet).
precision "ieee" (kernels.py) reproduces the arithmetic of stock TF 2.8.0 under its deterministic recipe (op determinism on, TF32 off, cuDNN algo 0)
bit for bit: each conv output is one serial fma.rn.f32 chain over r = c*K + k (input channel outer, tap inner) with the bias added after, and the
epilogue bias -> ReLU -> residual rounds once per step — the reduction order of cuDNN IMPLICIT_GEMM — written as an im2col tl.dot(input_precision='ieee').
Counts head (both sub-models): serial global average pool (positions summed in order, / L); the dense layer as 16 interleaved fma-chain lanes
(C=128: one chain per lane; C=512: two contiguous sub-chains per lane, channels < 256 and >= 256, summed per lane) combined sequentially (+ bias),
or the per-arch order named by arch_tiles.json counts_form; logcounts = reduce_logsumexp in TF's max-shifted form. precision "tf32" (kernels_tc.py)
runs the same convs on the tensor cores (stock's default precision class; not bitwise). Importing the package runs the typing_extensions guard
(_te_guard) before torch is imported.
"""
from ._te_guard import ensure_typing_extensions, TE_GUARD   # BEFORE any torch import in this package (see _te_guard)
ensure_typing_extensions()
from .kernels import K1BPNetV3, conv_dot, gap_serial, dense_exact, logsumexp_tf, exact_counts, DENSE_STRUCT, COMPILED
from .forward import K1ChromBPNet, featurize, predict, NARROWPEAK, INPUT_LEN, OUTPUT_LEN, apply, cache_witness, build_cache_record
