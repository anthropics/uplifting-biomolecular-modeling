# Third-party notices — AtlasFold kit

This kit vendors AtlasFold v1.0.0 unmodified under `stock/src` (its own `LICENSE` travels with it there) and adds an
optimization layer under `opt/`. The components below are third-party code that the kit's own files re-state or adapt, or
third-party code inside the vendored tree whose licence text upstream does not ship; the verbatim licence texts are under
`third_party_licenses/`. No file header inside `stock/` or `opt/` is added or changed by this notice. The kit's own code is licensed under
the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file concerns the third-party portions only.
Third-party code outside `stock/`: none is carried as a file of its own — every file outside `stock/` and `third_party_licenses/` was
written for this kit, and those that re-state upstream statements are named below. Data files: none outside `stock/`; the vendored tree
under `stock/src` holds AtlasFold's sources, configuration and documentation text (upstream's `docs/atlasfold.pdf` and `docs/images/` are
omitted, STOCK.md §Pin) and one data directory, `stock/src/assets/benchmarks/` — five FASTA files of evaluation target sequences as upstream
assembled them (`cameo22.fasta`: CAMEO targets named by wwPDB entry and chain, and `foldbench_abag.fasta`, `foldbench_pp.fasta`: FoldBench
complexes named by wwPDB entry and assembly — sequences of wwPDB entries, wwPDB archive data, CC0 1.0; `casp14.fasta`, `casp15.fasta`: target
sequences as published by the CASP Prediction Center) plus the identifier list `stock/src/assets/cameo_val_ids.txt`, carried unmodified as
part of the MIT-licensed release and never read by this kit. No weights, structures, sequence databases or alignments are in the tree. The example
inputs of the README are typed on the command line.

## AtlasFold — re-stated in the kit's lever modules

- Component: AtlasFold, https://github.com/SeonghwanSeo/atlasfold, tag v1.0.0 (commit `992067e67df29b665c501e0d2e9ead9dd4ba9b69`).
- Licence: MIT License. Copyright line as printed in the upstream `LICENSE`: "Copyright (c) 2026 Seonghwan Seo".
- Files concerned: the lever modules under `opt/atlasfold_opt/hooks/` each replace one AtlasFold function or method in the
  running process and re-state the parts of its body they leave unchanged (their docstrings cite the upstream file and
  lines), notably `sampler_hoist.py`, `sampler_hostsync.py`, `atom_kdedup.py`, `output_overlap.py`, `pair_block_residual.py`,
  `pair_transition_fused.py`, `pae_multimer.py`, `graph_reuse.py`, `exactln.py`; `opt/atlasfold_opt/phase_timing.py`,
  `registry.py` and `cli.py` describe those functions; the CPU tests `opt/atlasfold_opt/tests/test_sampler_hoist_cpu.py`,
  `test_sampler_hostsync_cpu.py`, `test_output_overlap_cpu.py`, `test_dit_apb_cpu.py` and `test_pair_fused_cpu.py` re-state
  upstream arithmetic as their reference.
- Licence text: `third_party_licenses/AtlasFold.MIT.txt` (byte-identical to `stock/src/LICENSE`). The MIT permission notice
  applies to these files' re-stated portions wherever `opt/` travels, with or without `stock/`.

## Notices inherited inside the vendored AtlasFold tree (files unchanged; texts supplied beside the tree)

The vendored tree carries these upstream files with their own headers and ships only the MIT text; the licence texts the
headers refer to are added here, outside `stock/`:

- `stock/src/src/atlasfold/model/network/primitives/initialize.py` — header: "Modified from OpenFold-3 initialize.py",
  "Copyright 2021 AlQuraishi Laboratory", "Licensed under the Apache License, Version 2.0". Component: OpenFold-3,
  https://github.com/aqlaboratory/openfold-3, Apache License 2.0. Text: `third_party_licenses/OpenFold3.Apache-2.0.txt`
  (the OpenFold-3 `LICENSE` as published).
- `stock/src/src/atlasfold/utils/checkpointing.py` — header: "Copyright 2026 Seonghwan Seo (KAIST), MIT LICENSE" and
  "Copyright 2025 AlQuraishi Laboratory (LICENSE-2.0)". Texts: `third_party_licenses/AtlasFold.MIT.txt`,
  `third_party_licenses/OpenFold3.Apache-2.0.txt`.
- `stock/src/src/atlaslm/layers/rotary.py` — header: "Started from the implementation of RotaryEmbedding in the ESM-3
  repository.", "Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.", Apache License 2.0 header.
  The Apache License 2.0 text in `third_party_licenses/OpenFold3.Apache-2.0.txt` is the same licence.
- `stock/src/src/atlasfold/common/ccd.py` — header: "Copyright Lenna Peterson (2012)", "This file is part of the Biopython
  distribution and governed by your choice of the "Biopython License Agreement" or the "BSD 3-Clause License"."
  Component: Biopython, https://biopython.org. Text: `third_party_licenses/Biopython.LICENSE.txt` (Biopython's `LICENSE.rst`
  as published in Biopython 1.85, which prints both the Biopython License Agreement and the BSD 3-Clause License).
- `stock/src/src/atlasfold/utils/rasa.py` — header: "Copyright 2026 Seonghwan Seo", "SPDX-License-Identifier: MIT"; text:
  `third_party_licenses/AtlasFold.MIT.txt`.

## Shared core

Kernels of the shared core (`../common/opt_core`) that this kit's levers call at run time — triangle multiplication,
triangle attention, transition, LayerNorm and pair-bias attention providers — are not part of this kit's tree; their
third-party notices and licence texts are kept beside them under `common/opt_core/` (per-directory `NOTICE` files and that
tree's third-party notices file).

## Model weights

The kit ships no weights. The three upstream Hugging Face repositories named in `STOCK.md` §Pin state the MIT License on
their model cards; `run.sh install --weights` only compares files on disk with the pinned digests.
