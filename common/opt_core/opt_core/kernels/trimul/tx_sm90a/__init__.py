# SPDX-License-Identifier: Apache-2.0
"""protenix_fpf_trimul_tx — Protenix v2 TriangleMultiplication (c_z = c_hidden = 256, bf16 autocast) for sm_90: hand-written CUDA prologue /
epilogue kernels (TMA + wgmma) around one cuBLAS strided-batched contraction.  Two providers with the stock module signature:
  protenix_fpf_trimul_tx.trimul:fn        FAST tier  (numerics class TOLERANCE: the cuEquivariance rounding points, LayerNorm statistics fp32 in the kernels'
                                          own summation order, padded planes)
  protenix_fpf_trimul_tx.trimul_exact:fn  EXACT tier (bitwise class: both LayerNorms inside K1 / K3 with a summation order chosen so results are bitwise
                                          identical to the stock op for supported shapes, projection /
                                          gating / residual arithmetic in the stock order, unpadded planes + the stock contraction expression; output ==
                                          the stock module's, bit for bit)"""
__version__ = "1.2"
