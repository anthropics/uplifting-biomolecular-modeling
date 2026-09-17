# LICENSES

| path | licence | origin |
|---|---|---|
| rfd_se3fast/dense_torch.py::str2str_forward_dense | contains a transcription of `Str2Str.forward` control flow from RosettaCommons/RFdiffusion `rfdiffusion/Track_module.py` @ 86507b6538f51fce57b5a72477165f03999ed7ae — BSD 3-Clause (VENDORED_FROM noted in the file header) | RFdiffusion |
| SE(3) layer semantics (ConvSE3 / AttentionBlockSE3 / basis layouts) re-implemented against NVIDIA SE3Transformer as vendored in RFdiffusion `env/SE3Transformer` @ the same commit — MIT | NVIDIA SE3Transformer |
| runtime dependencies (not vendored): torch (BSD), triton 3.0.0 (MIT), dgl 2.4.0 (Apache-2.0), e3nn 0.5.1 (MIT), hydra-core (MIT) | | |

RFdiffusion LICENSE (BSD 3-Clause; Copyright (c) 2023 University of Washington) and SE3Transformer LICENSE (MIT; Copyright 2021 NVIDIA CORPORATION & AFFILIATES) texts are carried verbatim in this tree: `rfdiffusion1/third_party_licenses/RFdiffusion.BSD-3-Clause.txt` and `rfdiffusion1/third_party_licenses/SE3Transformer.MIT.txt` (also `rfdiffusion1/stock/src/LICENSE` and `rfdiffusion1/stock/src/env/SE3Transformer/LICENSE`); the kit-level notices and per-file table are `rfdiffusion1/THIRD_PARTY_NOTICES.md`. Nothing else is vendored.
