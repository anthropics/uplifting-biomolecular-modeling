# Third-party notices — `esmfold2/`

This kit adapts, restates or wraps code of the third-party components below; the files concerned are listed per component.
The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file concerns the third-party portions only.
Licence texts that must travel with the adaptations are reproduced verbatim under `third_party_licenses/`; the vendored upstream sources under
`stock/` are unmodified and keep their own licence files. Per-directory notices with further detail stay beside the code:
`opt/forward/fast_inference/NOTICE`, `opt/forward/EF2_XL_ADDON_v1/NOTICE`, `opt/forward/fast_inference/tests/SOURCES.md`.

## `transformers`, Biohub fork (the ESMFold2 modeling code and its Triton kernels)

- Upstream: the Biohub fork of Hugging Face `transformers` (https://github.com/huggingface/transformers) at commit
  `ef32577f55da19a4989cd7b22e004dc43a4998cb` (the pin in `stock/PINS.json`). The fork's own repository, github.com/Biohub/transformers, is no
  longer publicly available. `stock/transformers-ef32577f.tar.gz` is a full source snapshot of the fork at that commit, made with `git archive`
  (recipe in `stock/PINS.json` "archive_recipe"): the complete `src/transformers/` package (every model directory, not only ESMFold2), `setup.py`,
  `pyproject.toml`, `README.md` and the fork's licence file `LICENSE` — 2,276 files under the prefix `transformers-ef32577f/`; `stock/src/transformers/`
  is a reading copy of 27 of them (byte-identical). The kit's environment (`environment/requirements.lock`) installs the same commit from
  https://github.com/huggingface/transformers, which serves the fork's commit unchanged; the pin check also accepts an install of the archive itself.
- Lineage, as recorded in the archive: `setup.py` and `src/transformers/__init__.py` declare version 4.57.6. Against the upstream release
  huggingface/transformers v4.57.6 the snapshot adds `src/transformers/models/esmfold2/` (24 files) and `src/transformers/models/esmc/` (6 files)
  and changes the four model-registry files `src/transformers/models/__init__.py`, `models/auto/configuration_auto.py`,
  `models/auto/modeling_auto.py` and `models/auto/tokenization_auto.py`; every other file is byte-identical to that release.
- Licence: Apache License 2.0 (the licence file inside the archive is `transformers-ef32577f/LICENSE`; the archive carries no `NOTICE` file)
- Copyright line, exactly as printed in the upstream `LICENSE`: `Copyright 2018- The Hugging Face team. All rights reserved.`. The fork's ESMFold2
  and ESMC source files that carry a header print `Copyright 2026 Biohub. All rights reserved.` / `Licensed under the Apache License, Version 2.0`;
  `models/esmfold2/modeling_esmfold2.py` and the seven kernel modules under `models/esmfold2/kernels/` (all but `kernels/__init__.py`) print no
  copyright line of their own. The eight files under `models/esmfold2/distributed/` carry `SPDX-License-Identifier: MIT` with
  `SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` and the MIT permission notice in full;
  the kit does not adapt them. The other model directories of the snapshot keep their own authors' Apache-2.0 (in a few files BSD- or
  MIT-style) headers, unchanged.
- Licence text in this tree: `third_party_licenses/transformers_fork.Apache-2.0.txt`, a byte-identical copy of the upstream `LICENSE` at that
  commit (the same text is `stock/src/transformers/LICENSE`).

| Kit file | Relation to the upstream code |
|---|---|
| `opt/forward/fast_inference/driver/ef2_w4.py` | modified copies of the Triton kernels of `src/transformers/models/esmfold2/kernels/` `fused_lnlin_swiglu.py`, `fused_dual_gemm.py`, `fused_ln_residual.py`, `trimul_with_residual.py` (inference-only variants, padded-layout addressing, row-block restructuring, tile tables; the modifications are marked in the file) |
| `opt/forward/fast_inference/driver/` `ef2_opt.py`, `ef2_pair_v2.py`, `ef2_transition_cute.py`, `ef2_xte.py`, `ef2_hoist.py`, `ef2_atom.py`, `ef2_dit.py`, `ef2_mk_sampler.py`, `ef2_msa.py`, `ef2_msa_v2.py`, `ef2_conf.py` | re-issue functions of `src/transformers/models/esmfold2/modeling_esmfold2.py` and `modeling_esmfold2_common.py` statement by statement with a kernel call, a cache or a hoisted constant substituted (the recycle loop, the MSA-encoder block, the pair transition, the triangle-multiplication block, the atom encoder / decoder, the diffusion sampler and transformer, the confidence head); the function list is `LEVER_SOURCES` in `opt/forward/fast_inference/driver/ef2_srcguard.py`, which refuses these modules by name when the installed upstream source differs from the pin |
| `opt/esmfold2_opt/` `rowpair.py`, `rowpair_heads.py`, `rowpair_msa.py`, `atom_swa.py`, `rowchunk/confrows.py`, `rowchunk/zbf16rows.py` | re-issue the same modeling code per row block of the pair representation for the multi-GPU route (`--n_gpu`); the copied statement ranges are cited in each module's docstrings |
| `opt/forward/EF2_XL_ADDON_v1/ef2_xl.py` | re-issues statement sequences of the model's forward for run-time memory patching; no upstream file is redistributed or modified on disk (see the `NOTICE` beside it) |
| `opt/forward/fast_inference/upstream/U1…U6_*.diff` | proposed patches against the fork at that commit; not applied by the kit |

## `esm` 3.3.0 (Biohub; model classes, input preparation)

- Upstream: https://github.com/Biohub/esm at commit `26b0bc2b771e3e419ea74f445a5f35cc094a1509` (the pin in `stock/PINS.json`).
  `stock/esm-26b0bc2b.tar.gz` is a `git archive` snapshot of that commit holding the `esm` package, `pyproject.toml`, the licence file
  `LICENSE.md` and upstream's `THIRD_PARTY_NOTICE.md` (114 files under the prefix `esm-26b0bc2b/`); `stock/src/esm/` is a reading copy of 30 of
  them (byte-identical). The archive's `pyproject.toml` describes the package as the "EvolutionaryScale open model repository" (authors:
  "EvolutionaryScale Team"), the code base the Biohub repository continues.
- Licence: MIT License (the licence file inside the archive is `esm-26b0bc2b/LICENSE.md`)
- Copyright line, exactly as printed in the upstream `LICENSE.md`: `Copyright 2026 Chan Zuckerberg Biohub, Inc.`. The package's source files print
  no per-file copyright line except two, which carry Apache-2.0 headers of other holders and are not adapted by the kit:
  `esm/layers/rotary.py` (`Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.`) and `esm/utils/residue_constants.py`
  (`Copyright 2025 EvolutionaryScale` / `Copyright 2021 AlQuraishi Laboratory` / `Copyright 2021 DeepMind Technologies Limited`). Upstream's
  `THIRD_PARTY_NOTICE.md` lists the libraries the package depends on with their licences; the data files the archive carries are listed under
  "Data and binaries" below.
- Licence text in this tree: `third_party_licenses/esm.MIT.txt`, a byte-identical copy of the upstream `LICENSE.md` at that commit (the same text
  is `stock/src/esm/LICENSE.md`)

| Kit file | Relation to the upstream code |
|---|---|
| `opt/forward/fast_inference/driver/ef2_feats.py` | vectorised re-implementation of `esm/models/esmfold2/paired_msa.py` `msa_to_res_type_and_deletions` and `construct_paired_msa` (byte-identical outputs), guarded by `ef2_srcguard.py` like the modules above |

## PyTorch 2.13.0 (one initialiser routine)

- Upstream: https://github.com/pytorch/pytorch at tag `v2.13.0` (the torch pin in `stock/PINS.json`; installed from the package index, not carried
  in this tree)
- Licence: BSD-3-Clause (PyTorch's `LICENSE`)
- Copyright lines, exactly as printed at the head of the upstream `LICENSE` (all of them are reproduced in the licence text below), beginning
  `Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)`; `torch/nn/init.py` itself prints no copyright line.
- Licence text in this tree: `third_party_licenses/PyTorch.BSD-3-Clause.txt`, a byte-identical copy of the upstream `LICENSE` at that tag.

| Kit file | Relation to the upstream code |
|---|---|
| `opt/esmfold2_opt/rowchunk/tn_shim.py` | `_no_grad_trunc_normal_` restates, modified, the rejection-sampling branch (acceptance mass above 0.3) of `torch/nn/init.py` `_no_grad_trunc_normal_`: the same signature, `norm_cdf` helper, bound casts and resampling loop, with the resample written in place; the low-acceptance branch and the range warning are omitted (the function raises instead). It is installed over `torch.nn.init._no_grad_trunc_normal_` only on a torch whose own routine is the inverse-CDF form |
| `opt/forward/EF2_XL_ADDON_v1/ef2_xl.py` | `_trunc_normal_lean_` (lever x8) restates the same branch with the out-of-range test and the `torch.where` selection issued per column block; the RNG calls are the upstream ones, at full size |

## Protenix (through the shared core)

- Upstream: https://github.com/bytedance/Protenix · Licence: Apache License 2.0 · Copyright line, exactly as printed in Protenix's `LICENSE`:
  `Copyright 2024 ByteDance and/or its affiliates.` (the shared core's kernel notices print it as `Copyright 2024 ByteDance and/or its affiliates`)
- The kit drives the shared core's `fpf_trimul_v4`, `fpf_transition` and `fpf_glue_v2` Triton kernels (`../common/opt_core/opt_core/kernels/`),
  portions of which restate the arithmetic of Protenix's pair-stack modules; `opt/forward/fast_inference/driver/ef2_w4.py`
  `_transition_fused_kernel` adapts the structure of the latter two. No Protenix source file is included in this directory. Notices and licence
  text: `../common/opt_core/opt_core/kernels/fpf_trimul_v4/NOTICE`, `fpf_transition/NOTICE`, `fpf_glue_v2/NOTICE`,
  `../common/opt_core/opt_core/kernels/fpf_flashpairformer.PROTENIX_LICENSE`.

## Data and binaries

- `opt/forward/fast_inference/tests/w4_public_slice.json`: three prediction inputs made of the amino-acid sequences of the two protein chains of
  PDB entry 1BRS (barnase / barstar) from the RCSB PDB — the 89-residue barstar chain and the 110-residue barnase chain as 1:1, 3:4 and 4:4
  copies; sequences only, no coordinates and no MSA (every `msa` field is `null`). PDB archive data are available under CC0 1.0 (wwPDB).
  Entry-level detail: `opt/forward/fast_inference/tests/SOURCES.md`.
- `stock/esm-26b0bc2b.tar.gz` carries, exactly as released by upstream under the archive's licence file, five data files under
  `esm-26b0bc2b/esm/data/` that belong to the package's function-annotation vocabulary: `entry_list_safety_29026.list` (an InterPro entry list:
  accession, type, name — the layout of InterPro's `entry.list`), `ParentChildTreeFile.txt` (InterPro's entry-hierarchy file of that name),
  and `interpro_29026_to_keywords_58641.csv`, `keyword_vocabulary_safety_filtered_58641.txt`, `keyword_idf_safety_filtered_58641.npy` (keyword
  tables keyed by InterPro accession — per-entry keywords, the vocabulary and its IDF weights — prepared by the upstream project). InterPro is
  EMBL-EBI's protein families resource (https://www.ebi.ac.uk/interpro/); its downloadable data are dedicated to the public domain under
  CC0 1.0. The archive does not record the InterPro release the files derive from or how the keyword tables were built, and upstream's
  `THIRD_PARTY_NOTICE.md` does not list them.
- `stock/transformers-ef32577f.tar.gz` carries source code only (Python modules, the CUDA / C++ kernel sources of other models, one Cython file,
  one shell script, two Markdown files): no data, weight or test-fixture file.
- No sequence database, alignment (MSA) or structure-coordinate file is carried anywhere under this directory.
- The kit ships no compiled object. `opt/forward/fast_inference/driver/prebuilt/sm_90a/` holds only `manifest.json` (an empty `"cubins"` set plus
  the toolchain record of the build route `driver/prebuilt/build_prebuilt.py`), a `SHA256SUMS` with no entry and `PROVENANCE.md`: no cubin or
  other binary, so no third-party object code (CUTLASS, CUDA or Triton output) is embedded there. The compute-capability-9.0 transition kernel
  object the kit binds at run time, `../common/opt_core/opt_core/kernels/transition/esm/ef2_t16/sm_90a/ef2_transition_cute.cubin`, was compiled
  against the NVIDIA CUTLASS (CuTe) and CCCL header wheels its manifest names; the object and the notices for those headers belong to the shared
  core (`../common/opt_core/opt_core/kernels/transition/NOTICE.md`).
- Model weights (ESMFold2, ESMFold2-Fast, ESMC-6B) are not part of this tree or of the images built from it: `run.sh install --weights DIR`
  fetches them from https://huggingface.co/biohub at the revisions pinned in `stock/PINS.json`; their model cards state their terms.
