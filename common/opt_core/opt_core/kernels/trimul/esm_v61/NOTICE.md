# NOTICE — `kernels.trimul.esm_v61` (row `esm_v61`; sealed payload `pkg/v6.1/`)

Licence: Apache License 2.0 (`common/opt_core/LICENSE`) for the files written in this repository; third-party portions as stated below.

Source. `__init__.py`, `cudrv.py`, `DESIGN.md` and every file of the sealed payload — `pkg/v6.1/python/face.py`,
`src/ef2_trimul_v5.py`, `src/ef2_trimul_v6.py` (which embeds the K3 kernel source as `K3_SRC`, byte-equal to `src/k3v6.cu`),
`src/k3v6.cu`, `src/PATCHES/k3v6_release_ordering_fix.diff`, `src/ef2_w4_fpf_trimul_v4_cells.json`, `tests/`, `CELLS.json`,
`bin/manifest.json`, `VERSION`, `SHA256SUMS` — were written in this repository and are licensed under the Apache License 2.0
(`common/opt_core/LICENSE`). No third-party source file is included. The K1 / K3 kernel structure derives from the
FlashPairformer TriangleMultiplication kernels carried at `kernels/fpf_trimul_v4` (this repository's code), whose operator
arithmetic follows the TriangleMultiplication module of Protenix — https://github.com/bytedance/Protenix, Apache-2.0,
Copyright 2024 ByteDance and/or its affiliates; licence text `kernels/fpf_flashpairformer.PROTENIX_LICENSE`. The batched
channel GEMM between K1 and K3 is one cuBLAS call of the installed stack.

Binaries. `pkg/v6.1/bin/k3v6.1_fastsig0_lnfold1.sm_90a.cubin` and `pkg/v6.1/bin/k3v6.1_fastsig1_lnfold1.sm_90a.cubin` were
compiled with NVRTC 13.0.88 from exactly `src/k3v6.cu` (source and cubin digests in `bin/manifest.json`) against the header
distributions `nvidia-cutlass` 4.2.0.0 and `nvidia-cuda-cccl` 13.0.85. `k3v6.cu` includes `<cute/tensor.hpp>` and instantiates
CuTe / CUTLASS templates, so both cubins contain object code derived from NVIDIA CUTLASS:

    CUTLASS / CuTe — https://github.com/NVIDIA/cutlass
    Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: BSD-3-Clause
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    conditions of the licence are met; the full text, including the disclaimer of warranty, is carried at
    common/opt_core/third_party_licenses/NVIDIA-CUTLASS.LICENSE.txt and applies to these two cubins.

The CUDA C++ Core Libraries headers used at build time (CCCL: Thrust / CUB / libcu++) are under the terms carried at
`common/opt_core/third_party_licenses/NVIDIA-CCCL.LICENSE.txt`. No CUTLASS or CCCL file is carried in this package. The cubins
are loaded through the CUDA driver API of the installed stack (`cudrv.py`); no CUDA toolkit component is included. K1 is a
Triton kernel compiled at first use by the installed Triton (MIT); nothing produced by Triton is shipped here.
