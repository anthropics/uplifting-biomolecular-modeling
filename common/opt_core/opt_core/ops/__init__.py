"""opt_core.ops — operations outside the kernel families (TriMul, triangle attention, attention with pair bias, transition, LayerNorm): modules moved
verbatim from the producing kits and served BY NAME by a kit's tier, each with its parity / bitwise evidence beside it.

Every op below raises ``opt_core.ops.FPFFallback`` (``_fallback``) for a call outside its served domain; the engine adapter catches it
and runs the stock module, counted.

  msa_opm    MSA outer-product mean: LayerNorm + a|b projections (+ mask) prologue, the outer-product GEMM over the MSA depth with the
             c_hidden^2 -> c_z projection (+ bias, / norm) fused into its epilogue -- the [N, N, c_hidden^2] tensor never reaches HBM.
  msa_pwa    MSA pair-weighted averaging (tolerance class): LayerNorm -> v | gate prologue, per-head softmax-weights contraction with the
             gate and the output projection in the epilogue.
  msa_pwa2   MSA pair-weighted averaging as an EXACT replica of the stock statements (bf16 autocast regime): candidate summation
             structures for the K = N contraction, selected per size class by a run-time bit-compare in the adapter.

Also here, imported by module path (no provider registration, no cell table; torch / triton imported by the modules themselves):

  atom_attn_window_exact/     the bit-exact windowed atom-attention op (Triton kernel + loadcheck + its GPU test)
  msa_fused/msa_triton.py     Triton kernels of the fused MSA-module ops (ln_linear, opm_out, pwa_ln_vg, pwa_out2), e.g. ``from opt_core.ops.msa_fused.msa_triton import ln_linear``

Related shared copies that live elsewhere in the core: the fused diffusion-transformer / atom-transformer Triton row kernels are the
pair-bias provider's carry (``opt_core.kernels.apb.ditfast.kernels`` / ``.atom_kernels``; row dit_fast, ``dit_block_rows()``), and the
CUDA-graph capture-safety instruments are ``opt_core.tools.graph_audit`` (tooling, not an operator).
"""
