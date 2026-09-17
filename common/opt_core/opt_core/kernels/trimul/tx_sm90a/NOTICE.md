# NOTICE — protenix_fpf_trimul_tx

SPDX-License-Identifier: Apache-2.0

All source in this package (`__init__.py`, `ops.py`, `trimul.py`, `trimul_exact.py`, `csrc/trimul_tx.cu`, `csrc/tx_ptx.h`) was written in this repository.
`csrc/tx_ptx.h` is a set of thin inline-`asm` wrappers over documented PTX instructions (mbarrier, cp.async.bulk.tensor, ldmatrix/stmatrix,
wgmma, setmaxnreg, cluster barriers) and the CUDA driver's `cuTensorMapEncodeTiled` entry point; `csrc/trimul_tx.cu` holds the two kernels
and their host launchers. No third-party source is included or adapted: no CUTLASS/CuTe headers, no Triton output, no vendor kernel code.
The triangle contraction between the two kernels is one `torch.bmm` / `torch.einsum` call, i.e. cuBLAS used as the installed library.  The exact-tier
LayerNorm returns results bitwise identical to those of the installed cuEquivariance LayerNorm for the supported shapes; it was written
independently, against that function's observable numerics only (inputs in, outputs compared), and a per-process self-check against the installed
function guards the equality at run time; no cuEquivariance source, intermediate representation or binary was read or copied for it.

The operator's arithmetic (rounding points, the order LayerNorm-then-gated-projection, sigmoid gating, residual add) follows the public
definition of the TriangleMultiplication module in Protenix (Apache-2.0) so that this provider is a drop-in for that module's forward.

Binary: `prebuilt/<abi tag>/protenix_trimul_tx_sm90.so` is compiled from exactly the `csrc/` files shipped here (digests in
`prebuilt/manifest.json`, build record in `prebuilt/<abi tag>/PROVENANCE.json`); it links against the CUDA driver/runtime and libtorch of the
stack only.
