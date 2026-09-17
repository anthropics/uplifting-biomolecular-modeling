# Third-party notices — `ef2inv/`

This kit adapts, restates or wraps code of the third-party components below and ships one compiled kernel object; the files concerned are
listed per component. The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file
concerns the third-party portions only. Licence texts
that must travel with the adaptations and with the binary are reproduced verbatim under `third_party_licenses/`; the vendored upstream
sources under `stock/` are unmodified and keep their own licence file. The kernel directory's own provenance record stays beside the code:
`opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_t16/PROVENANCE.md`.

## `transformers`, Biohub fork (the ESMFold2 and ESM-C model classes)

- Upstream: the Biohub fork of transformers at commit `ef32577f55da19a4989cd7b22e004dc43a4998cb` (fork version 4.57.6; the pin in
  `stock/PINS.json`; not carried in this tree). The fork's own repository, github.com/Biohub/transformers, is no longer public; the kit's
  environment installs that commit from https://github.com/huggingface/transformers, which serves it unchanged.
- Licence: Apache License 2.0
- Copyright line, exactly as printed in the upstream `LICENSE`: `Copyright 2018- The Hugging Face team. All rights reserved.`; the fork's ESMFold2 source files carry the header
  `Copyright 2026 Biohub. All rights reserved.` / `Licensed under the Apache License, Version 2.0`
- Licence text in this tree: `third_party_licenses/transformers_fork.Apache-2.0.txt`, a byte-identical copy of the upstream `LICENSE` at that
  commit. The upstream repository carries no `NOTICE` file.

| Kit file | Relation to the upstream code |
|---|---|
| `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_pairbias_attn.py` | replacement body for `AttentionPairBias.forward` of `modeling_esmfold2_common.py`; the prologue is the upstream statements verbatim (marked in the file), the attention call is the kit's |
| `opt/ef2inv_opt/patches.py` | `FUNCTION_TEXT` is a rewritten `Transition._addmm_residual` (the upstream lines it replaces are quoted in `STOCK.md`) |
| `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_autograd_kernels.py`, `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_trimul.py`, `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_trimul_nosave.py`, `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_esmc_rope.py`, `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_bf16_confidence.py`, `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_lazy_structure.py` | replacement bodies for further methods of the fork's ESMFold2 / ESM-C classes; each restates the fork's forward arithmetic around the changed lines and imports (does not copy) the fork's Triton kernels; bound at run time. The other `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_*.py` modules (CUDA-graph capture, checkpoint planning, scheduling, the kit's own kernels) wrap the fork's modules at run time without restating upstream statements. No upstream file is modified on disk. |

## `esm` (Biohub; the binder-design cookbook script the kit accelerates)

- Upstream: https://github.com/Biohub/esm at commit `d0207ea3c8cbf072679ece3bb332787d25e06852` (esm 3.4.0; the pin in `stock/PINS.json`;
  carried unmodified as `stock/esm-d0207ea3.tar.gz` and the reading copy `stock/src/`)
- Licence: MIT License. Copyright line, exactly as printed in the upstream `LICENSE.md`: `Copyright 2026 Chan Zuckerberg Biohub, Inc.`
- Licence text in this tree: `third_party_licenses/esm.MIT.txt`, a byte-identical copy of the upstream `LICENSE.md` at that commit (the same
  text is `stock/src/LICENSE.md`).

| Kit file | Relation to the upstream code |
|---|---|
| `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_loop_pppl.py` | re-issues `compute_esmc_pseudoperplexity_nll` of `cookbook/tutorials/binder_design.py` as a transcribed single step |
| `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_loop_prep.py` | re-plumbs `fold_and_get_distogram` of the same script |
| `opt/ef2inv_opt/settings.py` | reads the script's constants back; copies no code |

## Prebuilt binary this kit ships, and the headers it was compiled against

- `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_t16/sm_90a/ef2_transition_cute.cubin` (with `manifest.json`) is compiled from the kit's own kernel source
  `opt/forward/ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1/k/ef2_t16/ef2_transition_cute.cuh` for compute capability 9.0. The source includes NVIDIA CUTLASS / CuTe headers (`cute/tensor.hpp`),
  release 4.2.0, and the CUDA C++ Core Libraries headers of `nvidia-cuda-cccl` 13.0.85, as its manifest records.
- NVIDIA CUTLASS — https://github.com/NVIDIA/cutlass · BSD 3-Clause License · copyright line exactly as printed in its `LICENSE.txt`:
  `Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` · text: `third_party_licenses/NVIDIA-CUTLASS.LICENSE.txt` (clause 2 of that licence accompanies this binary redistribution).
- FlashAttention — https://github.com/Dao-AILab/flash-attention · BSD 3-Clause License · copyright line exactly as printed in its `LICENSE`:
  `Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.` · text: `third_party_licenses/Dao-AILab-flash-attention.LICENSE.txt`, carried together with the CUTLASS text for the release's CuTe-based
  compute-capability-9.0 kernel objects.
- The CUDA C++ Core Libraries (https://github.com/NVIDIA/cccl) are licensed under the Apache License 2.0 with LLVM exception; their licence
  text is not reproduced in this tree (the headers are used at build time only and no CCCL source is carried).

## Data files

None are carried, outside or inside `stock/`: the archive `stock/esm-d0207ea3.tar.gz` (43 members) holds upstream's Python sources, its
`README.md`, `LICENSE.md`, `pyproject.toml` and the cookbook script only — no structure, sequence, alignment, checkpoint or example output —
and the kit's own tree adds none (its tests build their inputs in code). The one binary file outside `stock/` is the compiled kernel object of
the previous section.

## Model weights

Not part of this tree or of the images built from it: `./run.sh install --weights DIR` fetches the ESMFold2 and ESMC-6B repositories from
https://huggingface.co/biohub at the revisions pinned in `stock/PINS.json`; their model cards state their terms (`license: mit`, with a pointer to
the upstream `THIRD_PARTY_NOTICE.md`, and the publisher's acceptable-use policy).

One third-party package of the pinned stack carries model files of its own: `anarcii` 2.0.8 (ANARCII, antigen-receptor sequence numbering; BSD
3-Clause, Copyright (c) 2025, University of Oxford; `environment/requirements.lock`) ships small numbering models inside the package —
`anarcii/classifii/classifii.pt` and `anarcii/models/{antibody,shark,tcr}/*_128_512.pt`, 2–4 MB each. They are installed with the stack on every route
(so they are present in the images) and are not part of this tree; their terms are the package's own licence.
