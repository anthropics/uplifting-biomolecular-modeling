# Third-party notices — `openfold3` kit

The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file concerns the
third-party portions only.

This kit carries OpenFold3 unmodified under `stock/` (the upstream `LICENSE` travels inside it, `stock/src/LICENSE`, and inside the wheel as
`openfold3-0.4.1.dist-info/licenses/LICENSE`) and adds an optimisation add-on under `opt/`. The add-on files listed below adapt or restate
third-party code. The kit ships no model weights. Verbatim licence texts are under `third_party_licenses/`.

## OpenFold3

- Component: OpenFold3 0.4.1 — https://github.com/aqlaboratory/openfold-3 (tag `0.4.1`, commit `d12f59554adacf8638e41f1ebc2d7725a8bb7a4b`).
- Carried as: `stock/openfold3-0.4.1-py3-none-any.whl` (the PyPI wheel of the tag) and `stock/src/` (the 504 files of the tagged source tree,
  byte for byte); `stock/PINS.json` records the pin.
- Licence: Apache License, Version 2.0. Upstream copyright line, as printed in the upstream LICENSE: "Copyright 2026 AlQuraishi Laboratory".
  The headers of a number of OpenFold3 source files print further copyright lines, quoted here as printed: "Copyright 2021 DeepMind
  Technologies Limited" (41 files: model layers, primitives, embedders, heads and latent stacks, the diffusion module, the alignment-tool
  wrappers under `core/data/tools/`, geometry and tensor utilities, two loss modules and `scripts/data_preprocessing/download_pdb_mmcif.sh`),
  "Copyright 2026 Advanced Micro Devices, Inc." (21 files), "Copyright 2025 NVIDIA Corporation" (`core/model/primitives/attention.py`,
  `core/model/latent/pairformer.py`, `core/model/latent/template_module.py`, `core/model/latent/base_stacks.py`) and "Copyright 2025
  AlQuraishi Laboratory" (`scripts/dev/update_ccd.py`, `scripts/snakemake_msa/test_download_of3_databases.py`). Every such file is under the Apache License, Version 2.0
  per its own header. The kit files below that restate statements of those files are modified works of them as well; the per-directory
  `NOTICE` files under `opt/forward/` name the lines that apply to each.
- Verbatim licence text: `third_party_licenses/OpenFold3.Apache-2.0.txt` (the same bytes as `stock/src/LICENSE` and as the `OPENFOLD3_LICENSE`
  copies beside the per-directory `NOTICE` files under `opt/forward/`).
- Kit files that restate or adapt OpenFold3 code (each says so in its own text; OpenFold3's files themselves are not edited):
  - `opt/forward/fast_inference/` — see `opt/forward/fast_inference/NOTICE` (`of3_levers/of3_fastinit.py`, `of3_levers/of3_graphs.py`).
  - `opt/forward/trunk_kernels/` — see `opt/forward/trunk_kernels/NOTICE` (`of3t_hook/of3t_levers.py`, `of3t_hook/of3t_paircache.py`; the
    directory also holds byte-identical copies of two OpenFold3 example inputs under `tests/inputs/`, see `tests/inputs/SOURCES.md`).
  - `opt/forward/offload/` — see `opt/forward/offload/NOTICE` (`of3o/of3o_confidence.py`, `of3o/of3o_blockreduce.py`, `of3o/of3_offload.py`).
  - `opt/openfold3_opt/of3_postfwd.py` — restates three OpenFold3 confidence functions (`compute_ptm` preamble and the `gpde` statements) per row block.
  - `opt/openfold3_opt/cells/of3_writer_overlap.py` — restates the output writer callback `on_predict_batch_end` so an item's files are written while the next item runs.
  - `opt/openfold3_opt/cells/of3_hostfeat.py`, `opt/openfold3_opt/cells/hostfeat.py` — restate host-side MSA featurisation statements of OpenFold3's data pipeline in numpy.
  - `opt/openfold3_opt/cells/postfwd_mem.py` — restates the post-forward confidence-scoring groups per (sample, row block).
  - `opt/openfold3_opt/cells/sync_hoist.py` — keeps the statements of `MSAModuleEmbedder.forward` / `_subsample_all_msa`
    (`core/model/feature_embedders/input_embedders.py`) and removes their host read-backs; the statements it serves are quoted in the module.
  - `opt/openfold3_opt/confhead.py` — restates the pair projections of `PredictedAlignedErrorHead.forward` / `PredictedDistanceErrorHead.forward`
    (`core/model/heads/prediction_heads.py`) per row block.
  - `opt/openfold3_opt/tp_rowpair/model.py`, `tp_rowpair/diffusion.py`, `tp_rowpair/data.py`, `tp_rowpair/confidence.py` — restate OpenFold3 0.4.1
    statements (`OpenFold3._rollout`, `SampleDiffusion.forward`'s loop and `centre_random_augmentation`, template featurisation lines,
    `compute_ptm`'s closing statements) for the multi-GPU row-sharded line (`big --n_gpu`).
  - `opt/openfold3_opt/tp_rowpair/trunk.py`, `tp_rowpair/msa.py`, `tp_rowpair/pairstack.py`, `tp_rowpair/template.py` — hand OpenFold3 0.4.1
    statements (`OpenFold3.run_trunk` and the `z_init` / recycling statements of `projects/of3_all_atom/model.py` and
    `feature_embedders/input_embedders.py`, `relpos_complex`, the MSA module's sub-module calls, the `PairBlock` / `PairFormerBlock` /
    `AttentionPairBias` / triangle-update projections, the template embedder) to the row-sharded drivers of the shared core
    (`../common/opt_core`) as callables, restating the statements each module's text names.
  These files are modified works of the OpenFold3 statements they name, distributed under the Apache License, Version 2.0; the changes are
  described in each file's module text. The other modules under `opt/openfold3_opt/` bind implementations of the shared core
  (`../common/opt_core`, which carries its own notices file) to OpenFold3's classes by name at import time.

## Third-party portions inside the carried OpenFold3 tree

The OpenFold3 files under `stock/` (source tree and wheel alike) mark, at the place of use, portions taken or adapted from other projects.
They are carried here exactly as upstream ships them:

- `openfold3/core/kernels/triton/swiglu.py` — functions from Liger-Kernel (https://github.com/linkedin/Liger-Kernel, `src/liger_kernel/ops/swiglu.py`
  and `ops/utils.py`; no commit is named). Liger-Kernel's licence is the BSD 2-Clause licence, "Copyright 2024 LinkedIn Corporation" as printed in
  its LICENSE; verbatim text: `third_party_licenses/Liger-Kernel.BSD-2-Clause.txt`. The same file cites a settings helper of
  https://github.com/unslothai/unsloth as a reference.
- `openfold3/core/kernels/triton/fused_softmax.py`, `triton_softmax.py` — "Taken from" FastFold (https://github.com/hpcaitech/FastFold,
  `fastfold/model/fastnn/kernel/softmax.py` and `kernel/triton/softmax.py`; no commit is named); FastFold's LICENSE is the Apache License,
  Version 2.0, "Copyright 2021- HPC-AI Technology Inc." as printed there, followed by the notices of the software FastFold itself incorporates;
  verbatim text: `third_party_licenses/FastFold.Apache-2.0.txt`.
- `openfold3/core/data/resources/patches.py` — a function "copied from" Biotite (https://github.com/biotite-dev/biotite,
  `src/biotite/structure/atoms.py`) and two Biotite functions re-written from Cython in Python; `scripts/data_preprocessing/preprocess_ccd_biotite.py`
  reproduces Biotite's BSD 3-Clause licence text in its header, "Copyright 2017, The Biotite contributors All rights reserved." as printed there.
- `openfold3/core/data/primitives/quality_control/logging_datasets.py` — a dataset class "Taken from PyTorch's ConcatDataset implementation"
  (https://github.com/pytorch/pytorch at commit `df458be4e5e96ce009ae1920da09a4095b34682e`; PyTorch's LICENSE at that commit is a BSD 3-Clause
  licence whose first copyright line reads "Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)"; verbatim text:
  `third_party_licenses/PyTorch.BSD-3-Clause.txt`); `openfold3/core/data/framework/lightning_utils.py` and
  `scripts/data_preprocessing/run_data_pipeline_treadmill.py` — a seed-sequence helper "Taken from Pytorch Lightning 2.4.1 source code"
  (https://github.com/Lightning-AI/pytorch-lightning at commit `f3f10d460338ca8b2901d5cd43456992131767ec`; its LICENSE is the Apache License,
  Version 2.0, "Copyright 2018-2021 William Falcon" as printed there; verbatim text: `third_party_licenses/PyTorch-Lightning.Apache-2.0.txt`).
- the MMseqs2-server client module under `openfold3/core/data/tools/` — the docstring of its server-query function (line 80 of that file)
  marks it "Adapted from" the MMseqs2 search function of the repository it links (github.com/sokrypton; MIT licence per that repository's
  LICENSE); `openfold3/core/metrics/rasa.py` — the relative-accessible-surface-area routine
  "Adapted from" https://github.com/BioComputingUP/AlphaFold-disorder (`alphafold_disorder.py`); `openfold3/core/data/resources/residues.py` — residue
  tables whose values are taken from Biopython's `Bio/Data/PDBData.py` (https://github.com/biopython/biopython).
- The files listed in the OpenFold3 section above whose headers print the DeepMind, Advanced Micro Devices and NVIDIA copyright lines.

No file under `stock/` names a copyleft licence (GPL, LGPL, AGPL, MPL, EPL, CDDL) or a non-commercial or share-alike restriction for code.

## Data files inside the carried OpenFold3 tree

Upstream's test data and examples travel inside `stock/src/` and, for the files under `openfold3/tests/test_data/`, inside the wheel as well
(same bytes). They are carried unchanged; none is read by the kit's own routes. Group by group:

- `openfold3/tests/test_data/mmcifs/*.cif` (10 files) — the wwPDB archive entries 1HF9, 1KD8, 1PSM, 2CRB, 2Q2K, 3U8V, 3ZEE, 4I6P, 4ZEY and 5KC1
  in PDBx/mmCIF format (`data_<ID>` blocks conforming to `mmcif_pdbx.dic`). PDB archive data are available under CC0 1.0 (wwPDB).
- `openfold3/tests/test_data/alignments/2q2k_A/`, `2q2k_B/` (5 files each; the two directories hold identical bytes) — upstream's test
  alignments for the 70-residue query sequence of PDB entry 2Q2K, i.e. the hit lists of sequence searches, excerpted as search tools write them:
  `uniref90_hits.sto` (Stockholm hits from UniRef90, `UniRef90_*` identifiers) and `uniprot_hits.sto` (hits from UniProtKB, `tr|…` identifiers) —
  UniRef and UniProtKB records of the UniProt Consortium, licensed CC BY 4.0 (https://www.uniprot.org/help/license), reproduced here as
  alignment rows with their description lines, otherwise unchanged; `mgnify_hits.sto` (hits from the MGnify protein database, `MGYP…`
  identifiers, EMBL-EBI; the MGnify release is not recorded in the file — current MGnify releases are CC0 1.0, earlier ones CC BY 4.0);
  `bfd_uniclust_hits.a3m` (HHblits-format hits of a BFD / Uniclust30 search; BFD (https://bfd.mmseqs.com) and Uniclust30 are licensed
  CC BY-SA 4.0 by their authors (Steinegger M. and Söding J.), and the hit rows carry UniProtKB accessions and description lines (UniProt
  Consortium, CC BY 4.0)); `hmm_output.sto` (an `hmmsearch` (HMMER 3.3.2, per the file's `#=GF AU` line) result of the query profile against PDB
  SEQRES sequences: chain identifiers, descriptions and aligned sequences of wwPDB entries, CC0 1.0).
- `openfold3/tests/test_data/template_alignments/inputs/` (`a3m_no_realign.a3m`, `a3m_realign.a3m`, `m8_cigar.m8`, `sto_hmmalign.sto`,
  `sto_hmmsearch_diff_seq.sto`, `sto_hmmsearch_same_seq.sto`) — small template-search results naming PDB chains (among them 7CNW_A, 7CNZ_A,
  2JSF_A, 4UT6_A, 6L06_A) with their SEQRES-derived aligned sequences (wwPDB, CC0 1.0); the query of `a3m_no_realign.a3m` is the first protein sequence of
  upstream's `examples/example_inference_inputs/query_multimer.json` (PDB 7CNX). `template_alignments/outputs/*.npz` (7 files) are the template
  arrays upstream's alignment parser produces from those inputs, stored as the tests' expected outputs.
- `openfold3/tests/test_data/tokenization/inputs/*_raw_bonds_unfiltered.npz` (5 files: 1EMA, 1PWC, 5SEB, 5TDJ, 6ZNC) — atom arrays parsed and
  sanitised from the wwPDB mmCIF files of those entries by upstream's parser, as `tokenization/README.md` beside them describes (source data
  CC0 1.0, wwPDB); `tokenization/outputs/*_tokenized_bonds_unfiltered.npz` (5 files) are the outputs of upstream's `tokenize_atom_array` on them,
  generated for the tests by `tokenization/construct_tokenization_examples.py` at the upstream change the README links.
- `openfold3/tests/test_data/permutation_alignment/inputs/npz/7pbd/7pbd_subset.npz` — an atom-array subset of wwPDB entry 7PBD (CC0 1.0);
  `outputs/npz/7pbd_subset_with-perm-ids.npz` — the same array with the permutation ids upstream's permutation alignment assigns, the test's
  expected output.
- `openfold3/tests/test_data/structure_from_query/*.pkl` (2 files) — pickled `StructureWithReferenceMolecules` objects produced by upstream's
  `structure_with_ref_mols_from_query` for the synthetic peptide sequence `MACHINELEARNING` (with and without two non-canonical residues), at the
  upstream commit `structure_from_query/README.md` names; generated fixtures, no database entry.
- `openfold3/tests/test_data/snapshots/test_triangular_attention/*.npz` (4), `test_triangular_multiplicative_update/*.npz` (2) and the two
  `_snapshot_env.json` files — expected-output tensors of upstream's triangle-attention and triangle-multiplication layer tests, recorded by
  upstream's test fixtures in the environment the JSON files state (torch 2.10.0+cu130, CUDA 13.0); generated numeric fixtures, no database content.
- `openfold3/tests/test_data/cassettes/test_rscb/*.yaml` (6 files) — recorded HTTP exchanges with the RCSB PDB Data API (`data.rcsb.org`, GraphQL
  and REST) for entries 1RNB and 4PQX and a non-existent id, replayed by upstream's tests offline; the response bodies are PDB archive metadata
  (chain-id mappings, ligand validation scores), CC0 1.0.
- `examples/example_inference_inputs/*.json` (7 files) — upstream's example queries, part of the Apache-2.0 tree: `query_multimer.json` (query
  `7cnx`: the two protein sequences of PDB entry 7CNX), `query_single_protein_single_ligand.json` (query `pdb_7L39`: a protein sequence and a
  ligand SMILES), `query_ubiquitin.json` (the 76-residue ubiquitin sequence), `query_homomer.json` (query `leucine_zipper`: a 34-residue
  homodimer), `query_protein_ligand.json` / `query_protein_ligand_multiple.json` (query `mcl1`: a protein sequence with the CCD ligand `ATP` and
  ligand SMILES strings) and `query_dna_ptm.json` (a 13-nucleotide DNA with two modified bases). Upstream states no source for these sequences
  beyond the query names; sequences of PDB entries are CC0 1.0 (wwPDB).
- `scripts/snakemake_msa/example_fasta_protein.fasta`, `example_fasta_RNA.fasta` — four example sequences named by PDB chain (6OR3_A, 7F6H_D,
  6I7O, 6SV4): SEQRES sequences of wwPDB entries, CC0 1.0.
- `openfold3/core/data/resources/canonical_protein_data/reference_molecule_data.json`, `reference_mols/*.sdf` (21 files) and
  `canonical_RNA_data/reference_molecule_data.json` — upstream's reference-conformer resources for the standard residues (SDF blocks written by
  RDKit, canonical SMILES per residue); generated resource files of the Apache-2.0 tree (in `stock/src/` only; the wheel does not package them).
- `docs/imgs/*.png`, `assets/*.png` — upstream's documentation images, part of the Apache-2.0 tree.

## Example inputs outside `stock/`

- `opt/forward/trunk_kernels/tests/inputs/query_7cnx_multimer_590tok.json`, `query_leucine_zipper_68tok.json` — byte-identical copies of
  `stock/src/examples/example_inference_inputs/query_multimer.json` and `query_homomer.json` (OpenFold3 0.4.1, Apache License, Version 2.0,
  "Copyright 2026 AlQuraishi Laboratory" as printed in the upstream LICENSE; the JSON files carry no header of their own), renamed by content;
  `tests/inputs/SOURCES.md` beside them states the entries.
- `opt/forward/fast_inference/tests/public_inputs.py` — embeds the two protein sequences of RCSB PDB entry 1BRS (barnase, 110 residues; barstar
  C40A/C82A, 89 residues) as string constants with their provenance line and a sha256 consistency check, and writes single-sequence inputs from
  them; nothing is fetched at run time. PDB entry data are CC0 1.0 (wwPDB).

## Python standard library (CPython) — `json` encoder

- Component: CPython standard library, `Lib/json/encoder.py` (`_make_iterencode`) — https://github.com/python/cpython.
- Licence: Python Software Foundation License Version 2 (PSF-2.0). Copyright notice, as printed in the licence: "Copyright (c) 2001, 2002, 2003,
  2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 Python Software Foundation;
  All Rights Reserved".
- Verbatim licence text: `third_party_licenses/Python.PSF-2.0.txt` (the CPython `LICENSE` file: history, the PSF License Agreement and the
  licences of incorporated software).
- Kit files concerned: `opt/openfold3_opt/cells/of3_fastjson.py` (bound by `opt/openfold3_opt/cells/fastjson.py`).
- Summary of changes (PSF-2.0 clause 3): the module contains a line-by-line port of `json.encoder._make_iterencode` (CPython 3.9–3.13) that
  emits into one list instead of yielding chunks, and renders 1-D / 2-D numeric arrays by joining `float.__repr__` / `int.__repr__` rows at
  C level; layout, separators, key order and coercion, `sort_keys` / `skipkeys` / `ensure_ascii` / `allow_nan` follow the standard encoder,
  and any structure the port does not reproduce is handed to the standard encoder unchanged.

## Software used at run time, not redistributed in this tree

PyTorch, PyTorch Lightning, DeepSpeed, Triton, NVIDIA cuEquivariance (`cuequivariance`, `cuequivariance-ops-cu12`, `cuequivariance-ops-torch-cu12`, `cuequivariance-torch`), NumPy,
Biotite and the other packages pinned in `environment/requirements.lock` are installed by the environment recipes under `environment/`
under their own licences; no file of theirs is copied under `openfold3/` beyond the portions inside the carried OpenFold3 tree listed above.
The model weights (`of3-p2-155k.pt`, OpenFold3 preview-2) are fetched at install time from the location `stock/PINS.json` names and are not
part of this tree; their terms are set by their provider.
