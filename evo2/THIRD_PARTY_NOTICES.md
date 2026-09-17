# Third-party notices — `evo2/`

This kit installs the two upstream packages below unmodified from the archives under `stock/` and binds its own functions over them at
run time; the files concerned are listed. The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file
concerns the third-party portions only. Licence and notice texts that must travel with the packages and with the code they contain are reproduced verbatim under
`third_party_licenses/`; the archives under `stock/` are unmodified (byte-identical to the files published on PyPI at the URLs in
`stock/PINS.json`), and the excerpt under `stock/src/` is those archives' members unpacked, byte for byte.

## Evo 2 (`evo2` 0.6.0)

- Upstream: https://github.com/arcinstitute/evo2, release 0.6.0 as published on PyPI (https://pypi.org/project/evo2/0.6.0/): the wheel
  `stock/evo2-0.6.0-py3-none-any.whl` and the sdist `stock/evo2-0.6.0.tar.gz` (sizes and PyPI URLs in `stock/PINS.json`).
- Licence: Apache License 2.0. Copyright lines, exactly as printed in the upstream `NOTICE`: `Copyright 2024 Arc Institute. All rights reserved`,
  `Copyright 2024 Michael Poli. All rights reserved`, `Copyright 2024 Stanford University. All rights reserved`. The same `NOTICE` states that the
  project incorporates and modifies components of StripedHyena (`Copyright 2023-2024 Together`), GPT-NeoX (`Copyright 2021-2024 EleutherAI and
  contributors`), Megatron-LM (`Copyright 2019-2024 NVIDIA Corporation`) and DeepSpeed (`Copyright 2020-2024 Microsoft Corporation`).
- Texts in this tree: `third_party_licenses/evo2.Apache-2.0.txt`, `evo2.NOTICE.txt`, `evo2.AUTHORS.txt` — byte-identical copies of the
  `LICENSE`, `NOTICE` and `AUTHORS` members of `stock/evo2-0.6.0-py3-none-any.whl` (`evo2-0.6.0.dist-info/licenses/`) and of the sdist.
  The `LICENSE` appendix reproduces the licences of portions from other organisations, as upstream names them: NVIDIA code (BSD 3-clause text,
  `Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.`), Hugging Face and Google Research code (Apache License 2.0) and Facebook
  Fairseq code (MIT License, `Copyright (c) Facebook, Inc. and its affiliates.`); those texts travel inside the same file.
- Data member: `evo2/test/data/prompts.csv` (in the wheel and the sdist; unpacked at `stock/src/evo2/test/data/prompts.csv`) — see "Data files" below.

## vortex (`vtx` 1.1.0, import name `vortex`; the inference engine Evo 2 runs on)

- Upstream: https://github.com/zymrael/vortex, release 1.1.0 as published on PyPI (https://pypi.org/project/vtx/1.1.0/): the wheel
  `stock/vtx-1.1.0-py3-none-any.whl` and the sdist `stock/vtx-1.1.0.tar.gz` (sizes and PyPI URLs in `stock/PINS.json`).
- Licence: Apache License 2.0. Copyright lines, exactly as printed in the upstream `NOTICE`: `Copyright 2024 Arc Institute. All rights reserved`,
  `Copyright 2024 Michael Poli. All rights reserved`, `Copyright 2024 Stanford University. All rights reserved`. The modules `vortex/model/cache.py`,
  `engine.py`, `generation.py`, `layers.py`, `model.py`, `vortex/ops/conv/csrc/step_fir.cu` and `vortex/ops/hyena_x/triton_indirect*_fwd.py` print
  the header `# Copyright (c) 2024, Michael Poli.`; the other vortex-authored modules print no copyright line.
- Texts in this tree: `third_party_licenses/vortex.Apache-2.0.txt` (the `LICENSE` member of the wheel and sdist) and `vortex.NOTICE.txt`
  (the `NOTICE` member of the sdist `stock/vtx-1.1.0.tar.gz`; the wheel the kit installs does not contain it, so it is supplied here).
- The wheel is tagged `py3-none-any`: it carries Python, C++ and CUDA sources only, no compiled object (`.so`, `.o`, `.cubin`). The sdist carries the
  same package files plus upstream's `test/` directory (Python test modules, no data files), `README.md`, `MANIFEST.in` and packaging metadata.

### Third-party code inside the vortex archives (wheel and sdist alike; upstream ships it, the kit does not modify it)

- FlashAttention (https://github.com/Dao-AILab/flash-attention; BSD 3-Clause License). Members: the FlashAttention 2 CUDA extension sources
  `vortex/ops/depr_attn/csrc/flash_attn/flash_api.cpp` and `vortex/ops/depr_attn/csrc/flash_attn/src/*` (kernel headers and the per-head-dimension
  `flash_fwd_*` / `flash_bwd_*` `.cu` instantiations) with `vortex/ops/depr_attn/setup.py` (first line: `modified from` the FlashAttention
  `setup.py`); the Python modules `vortex/ops/attn_interface.py`, `vortex/ops/embedding/rotary.py` and `vortex/model/rotary.py`. These files print
  `Copyright (c) 2023, Tri Dao.` or `Copyright (c) 2024, Tri Dao.` as their header. `vortex/model/positional_embeddings.py`, `layers.py`,
  `cache.py` and `sample.py` carry `Adapted from https://github.com/Dao-AILab/flash-attention/...` comments over the functions concerned.
  Licence text: `third_party_licenses/Dao-AILab-flash-attention.LICENSE.txt`, byte-identical to the `LICENSE` file of the FlashAttention
  repository at tag `v2.8.0.post2` (the flash-attn version pinned in `stock/PINS.json`); copyright line exactly as printed there:
  `Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.` The archives themselves carry no copy of this text.
- causal-conv1d (https://github.com/Dao-AILab/causal-conv1d; BSD 3-Clause License). Members: `vortex/ops/conv/csrc/causal_conv1d.cpp`,
  `causal_conv1d.h`, `causal_conv1d_common.h`, `causal_conv1d_fwd.cu`, `causal_conv1d_bwd.cu`, `causal_conv1d_update.cu` and
  `vortex/ops/conv/local_causal_conv1d/causal_conv1d_interface.py`, printing `Copyright (c) 2023, Tri Dao.` or `Copyright (c) 2024, Tri Dao.`;
  `vortex/ops/conv/setup.py` builds them as the `local_causal_conv1d` extension. Licence text:
  `third_party_licenses/Dao-AILab-causal-conv1d.BSD-3-Clause.txt`, a byte copy of the repository's `LICENSE` file (identical at every release tag
  and byte-identical to the FlashAttention text above); copyright line exactly as printed there:
  `Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.` The archives carry no copy of this text.
- NVIDIA CUTLASS 3.6.0 (https://github.com/NVIDIA/cutlass; `vortex/ops/cutlass/include/cutlass/version.h` defines `CUTLASS_MAJOR 3`,
  `CUTLASS_MINOR 6`, `CUTLASS_PATCH 0`, and the headers are byte-identical to the repository's tag `v3.6.0`). Members: the source tree under
  `vortex/ops/cutlass/` — `include/cutlass`, `include/cute`, `examples/`, `test/`, `tools/` and `python/` (1,852 files in the wheel; C++/CUDA
  headers and sources and Python, no data files and no compiled objects); the FlashAttention and causal-conv1d sources above include these
  headers. Every file prints `SPDX-License-Identifier: BSD-3-Clause` with the line `Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES.
  All rights reserved.` (some files print a first year of 2023 or 2024). Licence text: `third_party_licenses/NVIDIA-CUTLASS.LICENSE.txt`, a byte
  copy of `LICENSE.txt` of the CUTLASS repository at tag `v3.6.0` (the archives do not carry that file); copyright line exactly as printed there:
  `Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.`
- Single files with their own origin comments, as printed in them: `vortex/ops/conv/csrc/static_switch.h` and
  `vortex/ops/depr_attn/csrc/flash_attn/src/static_switch.h` (`Inspired by` NVIDIA DALI's `static_switch.h` and PyTorch's `ATen/Dispatch.h`);
  `vortex/ops/depr_attn/csrc/flash_attn/src/philox.cuh` (points at PyTorch's Philox implementation); `vortex/ops/depr_attn/csrc/flash_attn/src/generate_kernels.py`
  (`Copied from` a PyTorch pull request); `vortex/model/attention.py` (the ALiBi reference implementation in fairseq, and NVIDIA Apex / Megatron code);
  `vortex/model/sample.py` (NVIDIA Megatron-LM and Hugging Face `transformers` sampling utilities); `vortex/model/tokenizer.py` (`based on`
  EleutherAI GPT-NeoX's tokenizer module). The archives carry no separate licence text for these portions; the comments name the source files and commits.

## Kit files concerned

No upstream file is copied into `opt/` or `route/` and no upstream file is modified on disk. The kit's modules below bind their own functions
over methods of the installed `vortex` and `evo2` classes at run time, and the replacement bodies restate upstream statements (Apache License 2.0
material of the two packages above) around the changed lines; those files are therefore modified versions of the upstream functions named, and
are marked here as changed (Apache License 2.0, section 4(b)). No FlashAttention, causal-conv1d or CUTLASS code is restated in the kit's files.

| Kit file | Relation to the upstream code |
|---|---|
| `opt/evo2_opt/` and `route/` modules | bind the kit's own Triton kernels and library calls over methods of the installed `vortex` and `evo2` classes at run time; the replaced method bodies restate the upstream statements around the changed lines. No upstream file is modified on disk and no upstream source file is copied into `opt/`. |
| `opt/evo2_opt/kit/base.py` | restates statements of `vortex/model/engine.py` (`HyenaInferenceEngine.parallel_fir`, `parallel_iir`: the projection split, the FFT convolution and gating lines) and of `vortex/model/model.py` (`HyenaCascade.compute_filter`, `ParallelGatedConvBlock.forward`, `StripedHyena.stateless_forward`) around the kit's kernels and caches; modified. |
| `opt/evo2_opt/kit/kfft.py` | restates the point-wise spectrum-product Triton kernel and the gating epilogue of `vortex/ops/hcm_interface.py` and `vortex/ops/hcl_interface.py` (`hcm_fft_conv`, `hcl_fft_conv`) around cached filter spectra; modified. |
| `opt/evo2_opt/kit/hcs.py`, `kit/fir3rows.py` | restate the depthwise causal FIR tap loop of `vortex/ops/hcs_interface.py` (`_hcs_depthwise_conv_kernel`, `hcs_conv`) inside fused Triton kernels; modified. |
| `opt/evo2_opt/kit/compact.py`, `kit/c2r.py`, `kit/routes.py` | restate the projection-split and FFT-filter lines of `vortex/model/engine.py` (`parallel_fir`, `parallel_iir`) in the op chains they install; modified. |
| `opt/evo2_opt/kit/rotary.py` | replaces `LinearlyScaledRotaryEmbedding._update_cos_sin_cache` of `vortex/model/positional_embeddings.py` with a guard that returns early or calls the upstream function unchanged; no upstream statement restated. |
| `opt/evo2_opt/kit/batch.py` | restates `evo2/scoring.py` `prepare_batch` with the rows built as numpy arrays; modified. |
| `opt/evo2_opt/kit/pipelined.py`, `kit/pipeline.py` | wrap `Evo2.score_sequences` (`evo2/models.py`) and the two-device block loop of `StripedHyena.stateless_forward` (`vortex/model/model.py`) with upstream's signatures and exception handling restated; modified. |
| `opt/evo2_opt/activation.py` | defines the `Evo2` subclass bound at `import evo2` with the upstream constructor signature (`evo2/models.py` `Evo2.__init__`); no method body restated. |
| `opt/evo2_opt/gen/specdec/core.py`, `gen/specdec/__init__.py` | restate the logits clean-up and top-k / top-p statements of `vortex/model/sample.py` (`sample`) and the keyword handling of `Evo2.generate` (`evo2/models.py`) inside the speculative-decoding loop; modified. |
| `opt/evo2_opt/gen/cudagraph/__init__.py` | restates three lines of `StripedHyena.stateful_forward` and the `forward` signature of `vortex/model/model.py` around CUDA-graph capture and replay; modified. |
| `opt/evo2_opt/gen/hyenafuse/kernel.py`, `gen/hyenafuse/__init__.py` | re-express the per-token arithmetic of `HyenaCascade.sequential_forward` (`vortex/model/model.py`) and of the step functions in `vortex/model/engine.py` as one Triton kernel bound over each block instance; a re-implementation in another language of the same operations, no upstream statement copied. |

## Data files

- `evo2/test/data/prompts.csv` (member of both Evo 2 archives; unpacked copy `stock/src/evo2/test/data/prompts.csv`; 27,509 bytes): the input of
  upstream's generation self-test `evo2.test.test_evo2_generation` — a CSV with the columns `Sequence,Name,Percent` and four nucleotide prompts whose
  `Name` fields read `L1RE2`, `ECOLAC`, `NC_007596.2Mammuthusprimigeniusmitochondrion` and `NC_012920.1_homosapiens_mitochondrion`. `NC_007596.2`
  and `NC_012920.1` are NCBI RefSeq accession numbers (the woolly mammoth and the human mitochondrial genome records); `L1RE2` and `ECOLAC` are the
  names upstream gives the other two rows, and the archives state no accession or source for them. The file is distributed by upstream as package
  data under the package's Apache License 2.0; NCBI places no restrictions on the use or distribution of the sequence data in RefSeq and GenBank
  records (NCBI data usage policies). The kit's own tests and `route/` read no part of it.
- No other data file travels in this directory: the four archives contain source code, configuration (`evo2/configs/*.yml`, model
  hyper-parameters) and packaging metadata only — no structure files, alignments, arrays, pickles, checkpoints or expected-output fixtures — and the
  kit's tests under `opt/evo2_opt/tests/` build their inputs in code.

## Binaries and weights

- The kit ships no compiled object: its kernels are Triton source compiled at run time, plus calls into the cuFFT and cuBLASLt libraries that
  PyTorch loads. It does not import the shared core (`../common/opt_core`); `opt/_build_backend.py` is the kit's own copy of the shared core's
  build-backend template (the tree's own Apache-2.0 code, not third-party).
- Model checkpoints are not part of this tree or of the images built from it: `./run.sh install --weights DIR` fetches them from
  https://huggingface.co/arcinstitute at the revisions pinned in `stock/PINS.json`; their model cards state `license: apache-2.0`.
- The libraries the environment recipes install (PyTorch, Triton, flash-attn, Transformer Engine, the CUDA toolkit and cuDNN wheels) are fetched
  from their publishers' indexes at the versions in `environment/requirements-*.lock` under those publishers' own licences; none of them is
  carried in this directory.
