# Third-party notices — `enformer_deepmind`

This file lists the third-party components that this kit's own files wrap, reference or are compiled against, with the
licence of each and where its verbatim licence text is carried. The kit's own code lives under `opt/`; nothing under
`stock/` is edited. Model weights are not part of the kit (see the last section).

| component | upstream | version / commit | licence | how this kit relates to it | files in this kit concerned | verbatim licence text |
|---|---|---|---|---|---|---|
| Enformer model code and documentation (`enformer/` in deepmind-research) | https://github.com/google-deepmind/deepmind-research | commit `f5de0ede8430809180254ee957abf36ed62579ef`, directory `enformer/` | Apache-2.0; per-file headers read "Copyright 2021 DeepMind Technologies Limited" | documents the served model and its calling convention; no source file of the repository is included in or adapted by this kit — the kit rewrites the prediction graph of the published SavedModel in memory at run time | `stock/PINS.json` (pin and digests), `stock/LICENSE` (the repository-root `LICENSE` at that commit, unmodified) | `stock/LICENSE` and `third_party_licenses/deepmind-research.Apache-2.0.txt` (same bytes; sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`). The repository has no root `NOTICE` file at that commit. |
| TensorFlow | https://github.com/tensorflow/tensorflow | 2.17.1 (tag `v2.17.1`) | Apache-2.0; the included framework headers carry "Copyright 2015 The TensorFlow Authors. All Rights Reserved." | build-time headers: the prebuilt GPU custom-op library is compiled by `opt/enformer_deepmind_opt/ops/build.sh` from the kit's own sources (`ops/*.cc`, `ops/*.cu.cc`, `ops/edm_ops.h`, `ops/edm_f32.cuh`) against TensorFlow 2.17.1's C++ headers (`tensorflow/core/framework/op.h`, `op_kernel.h`, `shape_inference.h`) and links `libtensorflow_framework.so.2` dynamically; TensorFlow itself is not shipped (the user installs it from `environment/requirements.lock`) | `opt/enformer_deepmind_opt/ops/edm_ops.so` (596,856 bytes; build recorded by digest in `opt/enformer_deepmind_opt/ops/BUILD.json`) | `third_party_licenses/TensorFlow.Apache-2.0.txt` (the `LICENSE` file of tag `v2.17.1`, unmodified; sha256 `71c6915d04265772a0339bed47276942c678b45cc01534210ebe6984fd1aec65`). Tag `v2.17.1` has no root `NOTICE` file. |
| NVIDIA CUDA Toolkit (CUDA runtime headers, `nvcc`) | https://developer.nvidia.com/cuda-toolkit (PyPI wheels `nvidia-cuda-runtime-cu12` 12.3.101, `nvidia-cuda-nvcc-cu12` 12.3.107) | 12.3 (`nvcc` 12.3.107) | NVIDIA CUDA Toolkit End User License Agreement (PyPI licence field: "NVIDIA Proprietary Software") | build-time toolchain and headers: `edm_ops.so` is compiled with `nvcc` and includes `<cuda_runtime.h>`; the CUDA runtime library `libcudart.so.12` is linked dynamically and is not shipped with the kit (it arrives with the user's installed wheels) | `opt/enformer_deepmind_opt/ops/edm_ops.so` | not reproduced here (no NVIDIA library or header is redistributed); the agreement is published at https://docs.nvidia.com/cuda/eula/ |
| CUB (NVIDIA CCCL) | https://github.com/NVIDIA/cccl | CUB 2.2.0 | BSD-3-Clause | reference only: comments in `ops/edm_ops.cu.cc`, `ops/edm_softmax.cu.cc`, `ops/edm_layernorm.cu.cc` and `ops/README.md` cite CUB 2.2.0 (and TensorFlow 2.17.1) source locations as the reference for the floating-point reduction order the kernels reproduce; no CUB header is included and no CUB source is copied | none (citations in comments only) | not reproduced here (nothing of CUB is distributed) |

## Data files

None are carried. `stock/` holds only the pin record (`PINS.json`), its checker (`check_pins.py`) and the upstream repository's `LICENSE`;
no upstream source, SavedModel, sequence, genome interval, track or test fixture is in this kit, and the kit's tests build their inputs in
code. The one binary file in the tree is the custom-op library of the TensorFlow row above, compiled from the kit's own sources.

## Model weights

The Enformer SavedModel (`https://tfhub.dev/deepmind/enformer/1`) is not included in this kit and is not redistributed by
it: the user's own `tensorflow_hub.load(handle)` call obtains it from the publisher's hosting, or reads a cache directory the
user filled. `stock/PINS.json` records only the sha256 digests of its three files, which `stock/check_pins.py --weights`
compares. No terms for the weights are restated here; they are the publisher's.
