# Third-party notices — `opendde` kit

The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file concerns the
third-party portions only.

This kit carries OpenDDE 1.1.1 unmodified under `stock/` (the upstream wheel, sdist and source snapshot; upstream's `LICENSE` travels inside it as
`stock/src/LICENSE`) and adds an optimisation add-on under `opt/`. The add-on files listed below restate or adapt third-party code, and one
third-party extension is shipped in binary form together with its modified source file. The kit ships no model weights (`./run.sh install --weights DIR` fetches them through upstream's own
downloader). Verbatim licence texts are under `third_party_licenses/`. No header inside any source file is added or changed by this notice.

## OpenDDE

- Component: OpenDDE 1.1.1 — https://github.com/aurekaresearch/OpenDDE (tag `v1.1.1`, commit `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f`; PyPI `opendde` 1.1.1).
- Licence: Apache License, Version 2.0 (`stock/src/pyproject.toml`: `license = "Apache-2.0"`). The upstream `LICENSE` file carries no filled-in copyright
  line; the upstream source headers read "Copyright (c) 2026 Aureka AI Research" and, in files OpenDDE itself derives from other Apache-2.0 projects,
  "Copyright 2024 ByteDance and/or its affiliates." (Protenix), "Copyright 2021 AlQuraishi Laboratory" and "Copyright 2021- HPC-AI Technology Inc.",
  exactly as printed there.
- Verbatim licence text: `third_party_licenses/OpenDDE.Apache-2.0.txt` (the same bytes as `stock/src/LICENSE`).
- Kit files that restate or adapt OpenDDE code (each says so in its own text, naming the upstream function or file; OpenDDE's files themselves are not edited):
  - `opt/forward/fast_inference/levers/OFFLOAD/odde_offload_trunk.py` — the Pairformer / MSA-module block bodies of the 1.1.1 wheel with only tensor
    residency changed; `opt/forward/fast_inference/levers/OFFLOAD/odde_offload.py` — the triangle-multiplication inference forward and
    `OpenDDE.expand_to_structural_tokens`, host-resident.
  - `opt/opendde_opt/tp_struct.py` — the statements of `OpenDDE.expand_to_structural_tokens` before and after the expander call, for the row-sharded line;
    `opt/opendde_opt/tests/test_tp_struct_cpu.py` restates the same statements as the test's reference.
  - `opt/opendde_opt/chunklift.py` — `_get_dynamic_chunk_size`'s threshold table and `_bound_pairformer_chunk_size`;
    `opt/opendde_opt/tests/test_chunklift.py` restates the same two functions as the test's reference.
  - `opt/forward/fast_inference/levers/XL/odde_xl.py` — keeps the stock chunk-layer body in the un-chunked regime and restates the prologue (mask-bias)
    statements of `TriangleAttention.forward` (`opendde/model/triangular/triangular.py`); `opt/forward/fast_inference/levers/ARMT/odde_arm_t/__init__.py`
    (`make_castonce_ta_forward`) restates the body of the same `TriangleAttention.forward` with the LayerNorm output cast to bf16 once.
  - `opt/forward/fast_inference/levers/SAMPLER/third_party/opendde_fpf_ditfast/dit_fast.py` and `opt/forward/fast_inference/levers/DITFAST/tools/odde_addon.py`
    — a few statements of `AttentionPairBias.standard_multihead_attention` (pair-bias construction) from `opendde/model/modules/transformer.py`.
  - `opt/opendde_opt/templates.py` — one regular expression from `opendde/data/template/template_featurizer.py`.
  These files are modified works of the OpenDDE statements they name, distributed under the Apache License, Version 2.0; the changes are described in
  each file's module text.

## Prebuilt binary and its modified source — OpenDDE fused LayerNorm extension

- Files: `opt/forward/fast_inference/levers/LNSTREAM/prebuilt/torch2.7.1-cu126-cp311-cxx11abi1/fast_layer_norm_cuda_v2_cs.so` (compiled for sm_80 and sm_90;
  `manifest.json` and `SHA256SUMS` beside it describe and pin it) and `opt/forward/fast_inference/levers/LNSTREAM/src/layer_norm_cuda_kernel.cu` (the one
  modified source file, exactly as compiled). `opt/forward/fast_inference/levers/LNSTREAM/NOTICE` repeats this section beside them.
- Built from OpenDDE's own extension sources `opendde/model/layer_norm/kernel/` (in this tree: `stock/src/opendde/model/layer_norm/kernel/`) against
  PyTorch's C++ extension headers and the CUDA toolkit: `layer_norm_cuda.cpp`, `compat.h` and `type_shim.h` are compiled unmodified;
  `layer_norm_cuda_kernel.cu` is compiled with one change made by this kit — each of its 25 kernel-launch statements that name no CUDA stream is given
  the current stream (`<<<grid, block>>>` becomes `<<<grid, block, 0, at::cuda::getCurrentCUDAStream().stream()>>>`); device code, every other
  statement and the file's own notices are unchanged. `opt/opendde_opt/lnstream.py` (`to_current_stream`) makes the change,
  `python -m opendde_opt.lnstream prebuild` rebuilds the binary, and `opt/forward/fast_inference/levers/LNSTREAM/README.md` says when the binary is used.
- Licence of the four source files: Apache License, Version 2.0 (text: `third_party_licenses/OpenDDE.Apache-2.0.txt`). Their notices read, exactly as
  printed there: "Copyright 2024 ByteDance and/or its affiliates.", "Copyright 2020 The OneFlow Authors." and "Copyright 2021- HPC-AI Technology Inc."
  (`layer_norm_cuda_kernel.cu`); "Copyright 2021- HPC-AI Technology Inc." (`layer_norm_cuda.cpp`); "modified from
  https://github.com/NVIDIA/apex/blob/master/csrc/compat.h" and "modified from https://github.com/NVIDIA/apex", each followed by "Copyright 2021- HPC-AI
  Technology Inc." (`compat.h`, `type_shim.h`).
- NVIDIA apex portions: `compat.h` and `type_shim.h` are modified copies of NVIDIA apex files (https://github.com/NVIDIA/apex), as their headers state,
  and are compiled into the binary; the apex revision they were taken from is not recorded upstream. apex is distributed under the BSD 3-Clause
  licence — text: `third_party_licenses/NVIDIA-apex.BSD-3-Clause.txt`, and a second copy beside the binary,
  `opt/forward/fast_inference/levers/LNSTREAM/NVIDIA-apex.BSD-3-Clause.txt`; the upstream `LICENSE` prints no copyright line at its top (it opens
  "All rights reserved."). Its conditions and disclaimer apply to those portions of the source and of the binary.

## Licence and notice files inside the add-on directories

- `opt/forward/fast_inference/levers/ACCEL/NOTICE`, `levers/ARMT/NOTICE`, `levers/SAMPLER/NOTICE`, `levers/XL/NOTICE` and `levers/LNSTREAM/NOTICE` describe
  their directory's own files: the original code (Apache-2.0, the holder named in this kit's `LICENSE`), the portions derived from Protenix
  (Apache-2.0, ByteDance) through OpenDDE where a directory has any — `levers/ARMT` (the `TriangleAttention.forward` body restated by `odde_arm_t`; the pair
  `Transition` arithmetic followed by `third_party/fpf_transition_odde`), `levers/SAMPLER` (the `AttentionPairBias` statements and the
  DiffusionTransformer / AtomTransformer / DiffusionConditioning arithmetic re-implemented by `third_party/opendde_fpf_ditfast`), `levers/XL` (the
  `TriangleAttention.forward` prologue); `levers/ACCEL` restates no upstream statement — and the third-party software each calls.
- `opt/forward/fast_inference/levers/ACCEL/LICENSE`, `levers/ARMT/LICENSE`, `levers/SAMPLER/LICENSE`, `levers/XL/LICENSE` reproduce the Apache License,
  Version 2.0 text as distributed with Protenix (appendix line "Copyright 2024 ByteDance and/or its affiliates."), the project OpenDDE derives from: the
  licence of the derived portions named above and of the OpenDDE and Protenix functions these add-ons wrap at run time. They are not a separate grant
  over the original code, whose licence is this kit's `LICENSE`.
- `opt/forward/fast_inference/levers/SAMPLER/third_party/opendde_fpf_ditfast/NOTICE` is the notice of the shared core's fused row-kernel package
  (`opt_core.kernels.apb.ditfast` in `../common/opt_core`; original code of this release, Apache License 2.0), carried byte-identical with the core's
  copy; it names no kit. For this directory the facts are the `levers/SAMPLER/NOTICE` entry and the OpenDDE list above:
  `dit_fast.py` restates the few OpenDDE statements it names, the fused schedules re-implement the arithmetic of OpenDDE's diffusion-transformer
  blocks on the model's own weights, the rest is original code, and the row kernels it launches live in the shared core.

## Example inputs carried with upstream

- `stock/src/examples/` is OpenDDE's own `examples/` directory at the pin, unmodified: `2lwu.cif` and `7pzb.pdb` are the wwPDB entries 2LWU and 7PZB
  (wwPDB archive data, CC0 1.0; upstream ships them as template / input examples); `prot.fasta`, `dimer.fasta`, `example.json`,
  `example_without_msa.json`, `input.json`, `protein_200.json` and `foldcp_demo_placeholder.json` are upstream's example sequences and job files
  (distributed with the repository under its Apache-2.0 licence; `protein_200.json` and `foldcp_demo_placeholder.json` name themselves synthetic).
  The wheel and sdist under `stock/` carry no example or data files besides `opendde/config/model_manifest.json` (upstream's model catalogue).
  The kit adds no structure, sequence or alignment data of its own.

## Called, not included

PyTorch (BSD-3-Clause), Triton (MIT), NVIDIA cuEquivariance and the CUDA runtime (NVIDIA licences, as installed), RDKit and Biotite (as installed by
`environment/`), under their own licences as installed; none of their source or binaries is part of this tree except as stated above.
