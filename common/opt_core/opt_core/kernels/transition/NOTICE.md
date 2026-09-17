# NOTICE — opt_core/kernels/transition

The face (`__init__.py`) and the cell table (`TRANSITION_CELLS.json`) are this repository's own source.

Carried, byte for byte, from the kits of this release tree (each keeps its own header and licence position):

- `flash_sm90a/` — the `protenix_fpf_flash_transition` unit 1.1.0: `__init__.py`, `csrc/flash_transition_sm90.cu`, `csrc/build.py`,
  `cells/`, the prebuilt extension `prebuilt/torch2.13.0-cu130/flash_transition_sm90a.so` with its `manifest.json` (source and binary
  digests) and `PROVENANCE.md`, and its licence files `LICENSE.cutlass` (NVIDIA CUTLASS, BSD-3-Clause: CuTe headers used at build time),
  `LICENSE.flash-attention` (BSD-3-Clause) and `LICENSE_NOTE.md`. The operator's arithmetic follows the public definition of the Transition
  module in Protenix (Apache-2.0, https://github.com/bytedance/Protenix); the licence text is `kernels/fpf_flashpairformer.PROTENIX_LICENSE`.
- `esm/ef2_pair_v2.py`, `esm/ef2_autograd_kernels.py`, `esm/ef2_t16_transition.py`, `esm/ef2_t16_nvjit.py` — modules of the ESM-family kits
  (original source of this repository). The first two call, and do not include, the kernels vendored in the ESM-family `transformers` fork
  installed in that image. `esm/ef2_t16/` carries the CUDA C++ / CuTe kernel source `ef2_transition_cute.cuh` (original source), the sm_90a
  cubin built from it and the build manifest, with the design kit's `PROVENANCE.md`; the cubin was compiled against the NVIDIA CUTLASS (CuTe)
  and CCCL header wheels the manifest names (BSD-3-Clause and Apache-2.0-with-LLVM-exception headers used at build time; no NVIDIA source is
  included here).
- `af3/af3_fused.py` — a module of the AF3-PyTorch kit (original source of this repository, derived from `kernels/lnl_fused.py`).
- `esm_fused/__init__.py` — this repository's own source (row `esm_fused_exact`: the ESM-family fork's fused inference Transition statement in one
  Triton kernel). Six Triton device functions inside it are carried byte for byte from the ESM-family inference kit's `ef2_pair_v2.py`
  (`_ld_chunk`, `_bfly`, `_warp_tree`, `_stock_tree_sum`, `_stock_stats`) and `ef2_w4.py` (`_xhat_chunk`) — original source of this repository; the
  kernel calls, and does not include, PyTorch and OpenAI Triton, and reproduces the arithmetic of kernels vendored in the ESM-family `transformers`
  fork without including them.
Third-party software these rows call and do not include or modify: PyTorch (BSD-3-Clause), OpenAI Triton (MIT), the CUDA driver and runtime.
- `flash_prebuilt/<torch>-<CPython SOABI>-sm90/flash_transition_sm90a.so`: binaries of the sealed unit's UNCHANGED source (`flash_sm90a/csrc/`, digests in `flash_prebuilt/manifest.json` = the sealed manifest's) for other interpreter ABIs, built with the sealed recipe (`build_flash_prebuilt.py`); the sealed unit's LICENSE files and LICENSE_NOTE.md apply to them unchanged.
