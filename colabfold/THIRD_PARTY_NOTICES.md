# Third-party notices — ColabFold optimization kit (`colabfold/`)

This kit runs stock ColabFold 1.6.1 (`colabfold_batch`) with `alphafold-colabfold` 2.3.13 and changes their behaviour only at run time: classes and
functions are rebound inside the model process, and no upstream file is edited on disk. Both upstream packages are vendored unmodified under `stock/`
(the two PyPI wheels, each with its own LICENSE member, and reference copies under `stock/src/`). This file lists every third-party component that the
kit's own files adapt, restate or wrap, and the prebuilt binaries of the shared library (`../common/opt_core`) that the kit binds at run time. Verbatim
licence texts are in `third_party_licenses/`. The kit's own code is licensed under the Apache License 2.0 (LICENSE in this
directory and at the top of the tree); this file concerns the third-party portions only. The per-directory notice
`opt/forward/af2_pallas_flash/NOTICE` stays in place and points here.

## 1. Components adapted, restated or wrapped by the kit's own files

| Component | Upstream | Licence | Copyright line, as printed upstream | Kit files concerned | Verbatim licence text |
|---|---|---|---|---|---|
| AlphaFold 2, as shipped in `alphafold-colabfold` 2.3.13 (ColabFold's fork) | https://github.com/google-deepmind/alphafold · fork: https://github.com/sokrypton/alphafold | Apache License, Version 2.0 | `Copyright 2021 DeepMind Technologies Limited` (per-file headers, e.g. `alphafold/model/modules.py`, line 1) | `opt/colabfold_opt/templ_dedup.py` defines, at run time, a subclass of `alphafold.model.modules_multimer.TemplateEmbedding`; its `__call__` restates `alphafold/model/modules_multimer.py` lines 808-833 and 845-852 verbatim (marked in the file) and replaces the template scan body between them with the kit's own. `opt/colabfold_opt/device_resident.py`, `subbatch.py`, `trimul_pallas.py`, `msa_attn.py`, `msa_col_cudnn.py`, `transition.py`, `triattn_xla.py`, `big.py` and `opt/forward/af2_pallas_flash/af2_pallas_flash/af2_pallas_attn.py` rebind AlphaFold classes and functions (`RunModel`, `model_config`, `TriangleMultiplication`, `Attention`, `Transition`) at run time and describe the stock arithmetic they stand in for with line references; no upstream file is copied into them. | `third_party_licenses/AlphaFold2-DeepMind.Apache-2.0.txt` (byte-identical to `stock/src/alphafold/LICENSE` and to the wheel member `alphafold_colabfold-2.3.13.dist-info/LICENSE`) |
| ColabFold 1.6.1 | https://github.com/sokrypton/ColabFold | MIT License | `Copyright (c) 2021 Sergey Ovchinnikov` | `opt/colabfold_opt/_autoload.py` and `stack.py` wrap `colabfold.batch.run` at run time (the activation hook); `opt/colabfold_opt/settings.py` reads `colabfold_batch`'s option defaults from the installed `colabfold/batch.py` (parsed, not copied); `opt/colabfold_opt/weights.py` calls `colabfold.download.download_alphafold_params`; `opt/colabfold_opt/stock_pred.py` launches stock `colabfold_batch` unchanged. No ColabFold source is copied into the kit's own files. | `third_party_licenses/ColabFold.MIT.txt` (byte-identical to `stock/src/colabfold/LICENSE` and to the wheel member `colabfold-1.6.1.dist-info/LICENSE`) |

Modification notice (Apache License, Version 2.0, section 4(b)): the two excerpts restated in `opt/colabfold_opt/templ_dedup.py` are combined there with
this kit's own scan body (a template row identical to its predecessor reuses the predecessor's embedding); the upstream file `modules_multimer.py`
itself is not modified and is used as installed from the vendored wheel.

## 2. Prebuilt binaries of the shared library that this kit binds at run time

The kit ships no binaries of its own. Under `fast` and `big`, its `TRIATTN_XLA` lever (`opt/colabfold_opt/triattn_xla.py`) binds the shared library's
pre-compiled triangle-attention package `opt_core.kernels.triattn_xla`. Its binaries and the third-party material they contain or were compiled against,
as stated in `../common/opt_core/opt_core/kernels/triattn_xla/NOTICE` (the shared library's notices carry the full provenance):

| Binary, under `../common/opt_core/opt_core/kernels/triattn_xla/` | Third-party material | Licence | Verbatim licence text and copyright line |
|---|---|---|---|
| `bin/cuda/sm_90a/libtriattn_m1_xla.so` | compiled against the CUTLASS headers, release 4.7.1 (NVIDIA; no CUTLASS source is carried); its kernel header `fa3_utils.h` carries portions adapted from FlashAttention-3 (Dao-AILab) | BSD-3-Clause (both) | `third_party_licenses/NVIDIA-CUTLASS.LICENSE.txt` — `Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` · `third_party_licenses/Dao-AILab-flash-attention.LICENSE.txt` — `Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.` |
| `bin/cuda/sm_90a/libtriattn_mw_cuda.so`, `bin/cuda/sm_90a/libtriattn_mw_cuda_lse.so`, `bin/cuda/sm_80/libtriattn_sm80_xla.so` | the shared library's own CUDA sources per that NOTICE (the `sm_80` library: plain CUDA C++, no CUTLASS, no third-party source); each links the NVIDIA CUDA runtime statically | NVIDIA CUDA Toolkit EULA (runtime redistribution terms; referenced, not reproduced) | — |
| `bin/k2b/sm_80/*.cubin`, `bin/k2b/sm_90/*.cubin` | machine code produced by the Triton compiler and NVIDIA `ptxas` from the shared library's kernel source; portions of that kernel derive from Protenix | Triton: MIT License · Protenix: Apache License, Version 2.0 | the shared library's notices: `../common/opt_core/opt_core/kernels/fpf_triatt_k2b/NOTICE`, `../common/opt_core/opt_core/kernels/fpf_flashpairformer.PROTENIX_LICENSE` |
| `bin/launcher/ffi-*/libtriattn_xla_launch.so` | compiled against the XLA FFI API headers shipped in `jaxlib` (header-only use) and `cuda.h` | jaxlib: Apache License, Version 2.0 | the shared library's notices |

## 3. Shared-library source components this kit binds (attribution carried by the shared library)

- `../common/opt_core/opt_core/kernels/fpf_pallas/` (reached through the shared library's provider face for MSA column attention on compute capability 8.0):
  the shared library records that it extends AlphaFold 3 source (https://github.com/google-deepmind/alphafold3; the repository's LICENSE is the Apache License,
  Version 2.0; `Copyright 2024 DeepMind Technologies Limited`) and carries no upstream file. `../common/opt_core/opt_core/kernels/pallas_attn/`,
  `pallas_triatt/`, `pallas/cd_trimul/`, `pallas/cd_layers/`, `pallas/mlp_transition/` and `pallas/rowshared_flash_pallas.py` follow AlphaFold 2 formulas
  (Apache License, Version 2.0, `Copyright 2021 DeepMind Technologies Limited`) per their own NOTICE files; `../common/opt_core/opt_core/mem/rowpair_jax/alphafold.py`,
  `alphafold_template.py` and `alphafold_heads.py` (bound only under `big --n_gpu 2|4|8`) restate AlphaFold 2 method bodies over a row-sharded pair
  representation. The shared library's `THIRD_PARTY_NOTICES.md` and per-directory NOTICE files are the reference for these components; the AlphaFold 2
  licence text is also in this kit at `third_party_licenses/AlphaFold2-DeepMind.Apache-2.0.txt`.

## 4. Model parameters

The AlphaFold 2 parameters (`params_model_{1..5}_multimer_v3.npz`, DeepMind's 2022-12-06 release) are not included in this kit or in images built from it.
`./run.sh install --weights DIR` fetches them with ColabFold's own downloader into the user's directory and checks their digests (`stock/PINS.json`).
Their terms are the upstream's, stated by upstream as CC BY 4.0 (see `STOCK.md` and `opt/forward/af2_pallas_flash/NOTICE`); they are not restated here.

## 5. Third-party files carried inside the upstream wheels

| Component | Where in this kit | Licence | As printed upstream | Kit use |
|---|---|---|---|---|
| OpenStructure stereo-chemical properties table (`stereo_chemical_props.txt`), as packaged in ColabFold 1.6.1 | inside the carried wheel `stock/colabfold-1.6.1-py3-none-any.whl`, members `colabfold/openstructure/stereo_chemical_props.txt`, `colabfold/openstructure/LGPL.txt` and `colabfold/openstructure/README.md`; the same table, without a licence note, is also the member `alphafold/common/stereo_chemical_props.txt` of the carried wheel `stock/alphafold_colabfold-2.3.13-py3-none-any.whl` | GNU Lesser General Public License, Version 3 (LGPL-3.0) — the `LGPL.txt` member beside the file | `colabfold/openstructure/README.md`: "stereo_chemical_props.txt is part of openstructure and was downloaded from: https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt" and "OpenStructure is licensed under the GNU Lesser General Public License v3.0" | upstream ColabFold's own packaging, carried unmodified inside the wheel and installed with it at install time; no file of this kit reads, copies or adapts it |

## 6. Example inputs shipped with the kit

`tests/inputs/1BRS_AD.a3m` and `tests/inputs/1A3N_hemoglobin.a3m` (the README's example input and the `warm` verb's input) are query-only
complex inputs written for this kit: the paired query row plus one unpaired query row per chain, with no alignment database record in them.
Their sequences are those of wwPDB entries 1BRS (chains A and D, barnase and barstar) and 1A3N (chains A and B, human haemoglobin), wwPDB
archive data under CC0 1.0; `tests/README.md` says the same beside the input directory (kept out of `tests/inputs/` so that the
directory holds inputs only). No other structure, sequence or alignment file is
carried outside `stock/`, and the two upstream wheels carry none besides the OpenStructure table of section 5.
