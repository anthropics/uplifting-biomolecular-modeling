"""kernels.apb row ``atom_exact`` -- the carried EXACT-BITWISE fused local (atom-window) attention: kernel.py (one Triton launch per call
reproducing the windowed statement's bits: 32-query x 128-key windows at stride 32, pad mask + pair bias, warp softmax, TF32 / FFMA GEMM
statements as cuBLAS runs them for the chunked math-SDPA path), CELLS.json (its producing kit's cells), vectors.json (load-check digests),
NOTICE / README.md -- byte-identical to the producing kits' copies.  The kit-side binder (install.py: environment words, the trunk's module
patching) is not carried; the face (``opt_core.kernels.apb.atom_exact_attention``) runs the per-process route determination and the digest
load-check before the first launch and refuses by name where the statement's cuBLAS numerics are not reproducible."""
