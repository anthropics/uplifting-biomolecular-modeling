# Third-party notices — AlphaFold 3 (JAX) optimization kit

This file lists the third-party components that files of this kit adapt, restate, wrap or reach in binary form, with their
licences, their copyright lines exactly as printed upstream, the files concerned and where the verbatim licence texts sit
(`third_party_licenses/` beside this file). The vendored upstream tree under `stock/` keeps its own licence and notice files
as released (STOCK.md §Pin); nothing here replaces them. The AlphaFold 3 model parameters and the terms that govern them and
their outputs (`stock/src/WEIGHTS_TERMS_OF_USE.md`, `WEIGHTS_PROHIBITED_USE_POLICY.md`, `OUTPUT_TERMS_OF_USE.md`) are not
restated here: this kit ships, fetches and uses no AlphaFold 3 parameters (STOCK.md §Pin).

| component | upstream | licence | copyright line as printed upstream | verbatim licence text |
|---|---|---|---|---|
| AlphaFold 3 inference code | github.com/google-deepmind/alphafold3, through the fork github.com/sokrypton/alphafold3 tag `v3.1.4` (the pin, STOCK.md §Pin) | Apache License 2.0 | `Copyright 2024 DeepMind Technologies Limited` (head of every upstream source file, e.g. `stock/src/run_alphafold.py`; the LICENSE file itself prints no copyright line; the fork's added files — `convert_of3_weights.py`, `src/alphafold3/model/of3_weight_converter.py`, `src/alphafold3/data/msa_server.py`, `scripts/of3_verification/*.py` — print none and are covered by the fork repository's LICENSE) | `third_party_licenses/AlphaFold3-fork.Apache-2.0.txt` (byte-identical to `stock/src/LICENSE`) |
| tokamax | github.com/openxla/tokamax, release 0.0.12 (PyPI) | Apache License 2.0 | `Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.` (head of `tokamax/_src/ops/gated_linear_unit/pallas_triton.py` in that release; the release carries no NOTICE file) | `third_party_licenses/tokamax.Apache-2.0.txt` (the release's `LICENSE`) |
| CUTLASS (headers) | github.com/NVIDIA/cutlass | BSD-3-Clause | `Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` | `third_party_licenses/NVIDIA-CUTLASS.LICENSE.txt` |
| FlashAttention-3 (portions of one header) | github.com/Dao-AILab/flash-attention | BSD-3-Clause | `Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.` | `third_party_licenses/Dao-AILab-flash-attention.LICENSE.txt` |
| JAX / jaxlib (XLA FFI API headers) | github.com/jax-ml/jax | Apache License 2.0 | as printed in the jaxlib headers | header-only use at build time of the shared library's launcher (below); no jaxlib source is carried |
| Triton (compiler) | github.com/triton-lang/triton | MIT | as printed in the Triton repository | compiler whose output (`*.cubin`) the shared library ships (below); no Triton source is carried |

## Files of this kit that adapt or restate AlphaFold 3 code (Apache License 2.0, Copyright 2024 DeepMind Technologies Limited)

The files below restate or adapt function and module bodies of the AlphaFold 3 inference code named above. They carry no
upstream header of their own; this notice carries the attribution for them, and the licence text is
`third_party_licenses/AlphaFold3-fork.Apache-2.0.txt`.

- `opt/forward/fast_inference/patches/patched_files/run_alphafold.py` — a MODIFIED copy of the fork's `run_alphafold.py`
  (the upstream copyright and licence header is kept unchanged inside the file). The modifications are exactly
  `opt/forward/fast_inference/patches/01_run_alphafold_fast_inference.diff` (featurisation in worker processes, an output
  writer thread, one kernel-autotuning attempt per process, the kit's reporting lines); `opt/forward/fast_inference/HOWTO.md`
  describes them. Every kit mode runs this copy as `run_alphafold_fast.py`; `off` runs the fork's file.
- `stock/patches/04_of3_empty_template_restype_gap.diff` — nine added lines for `src/alphafold3/model/network/template_modules.py`,
  applied to the pinned source by `stock/unpack_src.sh` on every install route; the reading copy
  `stock/src/src/alphafold3/model/network/template_modules.py` is the upstream file with that patch applied (upstream header
  unchanged; STOCK.md §Stock exceptions states what the lines do).
- `opt/forward/flashpairformer/af3_flashpairformer/diffusion_hoist.py` rebinds `Model._sample_diffusion`,
  `DiffusionHead.__call__` and `Transformer.__call__`; `opt/forward/flashpairformer/af3_flashpairformer/patch.py` replaces
  `TriangleMultiplication` and `GridSelfAttention` with subclasses; the replaced bodies restate the upstream statements around
  the changed lines.
- `opt/forward/pallas_addon/patches/af3_pallas_levers.py` — a `TriangleMultiplication` subclass whose `__call__` is the
  upstream body with one branch replaced (its own notice: `opt/forward/pallas_addon/NOTICE`).
- `opt/af3_jax_opt/big_levers.py` and the in-process lever modules `opt/af3_jax_opt/inprocess/*.py` — rebind named Haiku
  modules of `alphafold3.model` (diffusion head, diffusion transformer, atom cross-attention, pairformer attention,
  transitions, LayerNorms, template embedding) and restate the upstream bodies they replace around the changed statements.
- Shared-library files this kit imports that do the same for AlphaFold 3 bodies: `common/opt_core/opt_core/mem/rowpair_jax/alphafold3.py`,
  `alphafold_template.py`, `alphafold_heads.py` (row-sharded restatements of the pair stack, template embedder and heads) and
  `common/opt_core/opt_core/kernels/fpf_pallas/*`, `fpf_pallas_serve.py` (Pallas kernels that extend the AlphaFold 3 triangle
  operations). Their notices are kept with the shared library (`common/opt_core/`).

## tokamax (Apache License 2.0)

- `opt/forward/pallas_addon/patches/af3_pallas_levers.py` (`glu_transposed_masked`) and the shared library's
  `common/opt_core/opt_core/kernels/pallas_glut.py` re-implement the Pallas-Triton gated-linear-unit kernel of
  `tokamax/_src/ops/gated_linear_unit/pallas_triton.py` (release 0.0.12) with a transposed and masked epilogue. Notices beside
  the code: `opt/forward/pallas_addon/NOTICE`, `common/opt_core/opt_core/kernels/pallas_glut.NOTICE`. Licence text:
  `third_party_licenses/tokamax.Apache-2.0.txt`.

## Prebuilt binaries this kit reaches (shared library package `opt_core.kernels.triattn_xla`)

The `fast` and `big` modes import `opt_core.kernels.triattn_xla`, whose `bin/` directory ships compiled code. That package's
`NOTICE` (`common/opt_core/opt_core/kernels/triattn_xla/NOTICE`) states the provenance of every part; the third-party
conditions that travel with binary redistribution are met here by the texts under `third_party_licenses/`:

- `bin/cuda/sm_90a/libtriattn_m1_xla.so` — built against the CUTLASS headers (BSD-3-Clause, NVIDIA; release 4.7.1, no CUTLASS
  source carried) and including `fa3_utils.h`, which carries portions adapted from FlashAttention-3 (BSD-3-Clause; the notice is inside that header, carried at
  `common/opt_core/opt_core/kernels/triattn/triattn_native/pkg/v11/triattn_pkg/cuda_b/csrc/fa3_utils.h`). Texts: `third_party_licenses/NVIDIA-CUTLASS.LICENSE.txt`,
  `third_party_licenses/Dao-AILab-flash-attention.LICENSE.txt`.
- `bin/cuda/sm_90a/libtriattn_mw_cuda.so`, `libtriattn_mw_cuda_lse.so`, `bin/cuda/sm_80/libtriattn_sm80_xla.so` — the shared
  library's own CUDA sources compiled with nvcc; statically linked NVIDIA CUDA runtime (see that package's NOTICE).
- `bin/k2b/sm_80/*.cubin`, `bin/k2b/sm_90/*.cubin` — machine code produced by the Triton compiler (MIT) and NVIDIA ptxas from
  `common/opt_core/opt_core/kernels/fpf_triatt_k2b/triatt_k2b.py`; portions of that kernel derive from Protenix
  (Copyright 2024 ByteDance and/or its affiliates, Apache License 2.0 — `common/opt_core/opt_core/kernels/fpf_triatt_k2b/NOTICE`,
  licence text `common/opt_core/opt_core/kernels/fpf_flashpairformer.PROTENIX_LICENSE`).
- `bin/launcher/ffi-*/libtriattn_xla_launch.so` — compiled from the package's own `csrc/cubin_launch.cc` against the XLA FFI API
  headers shipped in jaxlib (Apache License 2.0, header-only use) and `cuda.h`.

## Data files

- Outside `stock/`: `opt/forward/fast_inference/tests/inputs/1brs_barnase_barstar.json` — an AlphaFold 3 input holding the two protein
  sequences of wwPDB entry 1BRS (barnase, barstar; wwPDB archive data, CC0 1.0) with empty alignment and template fields. No other data
  file sits outside `stock/`; `opt/af3_jax_opt/kit_required_files.json` and `opt/af3_jax_opt/tests/printed_lines.json` are this kit's own
  file lists and expected report lines.
- Inside `stock/alphafold3-bc32b22f.tar.gz` only (not in the reading copy `stock/src/`), unmodified, the upstream package's test data
  `src/alphafold3/test_data/`, which no file of this kit reads:
  - `miniature_databases/` — 1,000-sequence excerpts of the genetic databases AlphaFold 3 searches, included by upstream "with the
    inference code package for testing purposes" under the terms its README states (`stock/src/README.md`, 'Mirrored and Reference
    Databases', quoted there in full with versions and citations): `bfd-first_non_consensus_sequences__subsampled_1000.fasta` — BFD as
    modified by Google DeepMind (the reduced 'small BFD'), CC BY 4.0 (the unmodified BFD itself is distributed by its authors under
    CC BY-SA 4.0; this excerpt is DeepMind's modified form under the terms the README prints); `mgy_clusters__subsampled_1000.fa` — MGnify
    2022_05, CC0 1.0; `uniprot_all__subsampled_1000.fasta` — UniProt 2021_04, CC BY 4.0; `uniref90__subsampled_1000.fasta` — UniRef90
    2022_05, CC BY 4.0; `pdb_seqres_2022_09_28__subsampled_1000.fasta` and `pdb_mmcif/{5y2e,6s61,6ydw,7rye}.cif` — wwPDB sequences and
    entries, CC0 1.0; `rfam_14_4_clustered_rep_seq__subsampled_1000.fasta` — Rfam 14.4 (modified), CC0 1.0;
    `rnacentral_active_seq_id_90_cov_80_linclust__subsampled_1000.fasta` — RNAcentral 21.0 (modified), CC0 1.0;
    `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq__subsampled_1000.fasta` — the NCBI nucleotide collection (NT 2023_02_23, modified;
    the README prints its citation and no licence line; NCBI distributes it without restriction).
  - `alphafold_run_outputs/run_alphafold_test_output_bucket_{1024,default}.pkl`, `featurised_example.pkl`, `featurised_example.json`,
    `model_config.json` — the featurised test input and the expected-output pickles of upstream's `run_alphafold_test.py`, generated
    by upstream and shipped in its Apache-2.0 source package for its own tests.
  A wheel built from the source by the routes of STOCK.md §Stack contains what upstream's build includes; this kit adds nothing to it.
- No AlphaFold 3 parameters, no OpenFold3 checkpoint and no full genetic database are in this tree (STOCK.md §Pin says what
  `run.sh install --weights` fetches).

The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this
file concerns the third-party portions only.
