# ef2_t16/ — the carried t16 pair-transition forward kernel (source, shipped sm_90a cubin, manifest)

The kernel source, its prebuilt cubin and the cubin's build manifest travel together, so that this copy stays self-contained (no
import across packages at run time):

| file here | sha256 |
|---|---|
| `ef2_transition_cute.cuh` | `1d6ea12897078d58b1414076c440c9f51abc5a2acdaa281c100b2419ea184446` |
| `sm_90a/ef2_transition_cute.cubin` | `2f600bd17b86cbee53cc0509c3a9238e5ce7ca5a97158e1543089d95f4077af3` (also in `sm_90a/SHA256SUMS`) |
| `sm_90a/manifest.json` | `67734e22937641a244c945a4af3827a6ba4598697b3d72308361520b01305fcc` |

Beside this directory, `../ef2_t16_nvjit.py` is the loader (its constants PREFIX, SHIPPED_ROOT → this directory, PREBUILD_MODULES →
`ef2_t16_transition`; it holds the shipped cubin to `sm_90a/SHA256SUMS`) and `../ef2_t16_transition.py` carries the kernel-side statements
(VERSION, constants, `kernel`, the canary, `pack_transition`, `_tmx`, `grid_for`, `transition_cute`, the references).

The cubin: built by the loader's `build_all` with NVRTC 13.0 from the header wheels its manifest names (nvidia-cutlass 4.2.0.0,
nvidia-cuda-cccl 13.0.85), `--gpu-architecture=sm_90a`; ptxas 168 registers, 0 B stack, 0 spill; 230512 B dynamic shared memory;
canary digest `6a3438b92fec…` recorded on an H100 80GB HBM3. Nothing here compiles it at run time: `ef2_t16_nvjit` serves the shipped file
when its sha256, source key (the `.cuh` text + compile options, computed here without a toolchain), spill record and the device-reported
register / local-memory counts all match, else the lever steps aside by name and K-D3's own forward kernels serve the pair transition.
A rebuilt cubin is carried the same way: the three files above replaced together, this table's digests with them.
