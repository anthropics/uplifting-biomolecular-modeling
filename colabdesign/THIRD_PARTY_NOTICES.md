# Third-party notices — `colabdesign/`

This kit carries two upstream source archives under `stock/` (members unmodified; the BindCraft archive without its two prebuilt
executables, see *BindCraft* below) with reference copies under `stock/src/`, imports BindCraft's
design step from them at run time, and re-expresses parts of ColabDesign and of the AlphaFold 2 model code it contains inside the kit
package `opt/colabdesign_opt/`. Every third-party component involved is listed below with the files concerned and the place where its
verbatim licence text sits in this tree (`third_party_licenses/`). The vendored copies under `stock/` are never edited; licence texts
that are missing inside an upstream archive are supplied beside it here, not inserted into it. No AlphaFold parameters are included in
this tree or in the container images built from it: `run.sh install --weights DIR` fetches the five AlphaFold-Multimer v3 parameter
files into `DIR/params/` (their terms: STOCK.md, Licences). The shared library's kernel notices stay beside the kernels
(`common/opt_core/opt_core/kernels/**/NOTICE`).

## ColabDesign

- Upstream: https://github.com/sokrypton/ColabDesign, version 1.1.3 at commit `e31a56fe1d9b4de25c8697f3a28b75892941cc72` (the pin in `stock/PINS.json`)
- Licence: `"THE BEER-WARE LICENSE" (Revision 42)` — the title line of the upstream `LICENSE.txt`; the notice names Sergey Ovchinnikov as the author
  and asks that the notice be retained
- Licence text in this tree: `third_party_licenses/ColabDesign.Beerware.txt`, a byte-identical copy of the upstream `LICENSE.txt` at that commit
  (the same bytes are carried at `stock/src/LICENSE.txt` and inside `stock/colabdesign-e31a56fe.tar.gz`)

| Kit file | Relation to the upstream code |
|---|---|
| `stock/colabdesign-e31a56fe.tar.gz` | `git archive` of `colabdesign/ setup.py LICENSE.txt README.md` at the pinned commit, carried unmodified |
| `stock/src/LICENSE.txt`, `stock/src/README.md`, `stock/src/setup.py`, `stock/src/colabdesign/**` | unmodified copies of archive members, carried for reading and for the pin check's digests |
| `opt/colabdesign_opt/hoist_prev.py` | restates the recycle loop of upstream `colabdesign/af/design.py` (`_af_design._recycle` and `run`) with the recycle-0 input built on the device; installed over the class at run time under a kit mode |
| `opt/colabdesign_opt/nosub.py`, `nosub_fn.py`, `lowercache.py`, `kernels/*.py`, `pallas.py`, `txla.py` | wrap or rebind ColabDesign functions and haiku module classes at run time (sub-batching rule, model build, attention / triangle-multiplication / Transition / LayerNorm / outer-product-mean call sites); no ColabDesign source file is copied into `opt/` |

## AlphaFold 2 model code (inside ColabDesign, `colabdesign/af/alphafold/**`)

- Upstream: https://github.com/google-deepmind/alphafold (the AlphaFold 2 source that ColabDesign includes under `colabdesign/af/alphafold/`)
- Licence: Apache License, Version 2.0
- Copyright line, exactly as printed in the headers of those files: `Copyright 2021 DeepMind Technologies Limited`
- Licence text in this tree: `third_party_licenses/AlphaFold.Apache-2.0.txt`, a byte-identical copy of the `LICENSE` file of the AlphaFold
  repository (release v2.3.2); the ColabDesign archive itself carries the file headers but not this text

| Kit file | Relation to the upstream code |
|---|---|
| `stock/colabdesign-e31a56fe.tar.gz` (members `colabdesign/af/alphafold/**`), `stock/src/colabdesign/af/alphafold/model/modules.py` | unmodified upstream files as released inside ColabDesign |
| `opt/colabdesign_opt/kernels/trimul_fused.py`, `kernels/triatt_lever.py`, `kernels/proj_attn.py`, `kernels/layers_ln.py`, `kernels/layers_opm.py`, `kernels/layers_transition.py`, `pallas.py`, `txla.py` | re-express the arithmetic of the AlphaFold 2 modules `Attention`, `TriangleMultiplication`, `Transition`, `LayerNorm` and `OuterProductMean` for fused execution and hand the calls to the shared library's kernels (`common/opt_core/opt_core/kernels/pallas*`, `triattn_xla`, each with its own NOTICE); no AlphaFold source file is copied into `opt/` |

## ESM — MSA Transformer modules (inside ColabDesign, `colabdesign/esm_msa/*`)

- Upstream: https://github.com/facebookresearch/esm (the modules ColabDesign includes under `colabdesign/esm_msa/`; their headers read
  `Copyright (c) Facebook, Inc. and its affiliates.` / `Levinthal, Inc. and its affiliates` and refer to an MIT `LICENSE` file that the
  ColabDesign archive does not carry; `esm_msa/config.py` is headed Apache License 2.0, `Copyright 2021 Levinthal Limited`)
- Licence: MIT License
- Copyright line, exactly as printed in the upstream `LICENSE`: `Copyright (c) Meta Platforms, Inc. and affiliates.`
- Licence text in this tree: `third_party_licenses/ESM.MIT.txt`, a byte-identical copy of the upstream `LICENSE`; the Apache License 2.0 text
  is `third_party_licenses/AlphaFold.Apache-2.0.txt`
- Kit files concerned: members of `stock/colabdesign-e31a56fe.tar.gz` only; the kit's design step does not import these modules

## ProteinMPNN parameters (inside ColabDesign, `colabdesign/mpnn/weights/*.pkl`, `colabdesign/mpnn/weights_soluble/*.pkl`)

- The carried upstream archive `stock/colabdesign-e31a56fe.tar.gz` contains 8 ProteinMPNN parameter files (`colabdesign/mpnn/weights/*.pkl`
  and `colabdesign/mpnn/weights_soluble/*.pkl`) as released by ColabDesign, together with ColabDesign's own `colabdesign/af/weights/template_dgram_head.npy`;
  the same files are present wherever the pinned ColabDesign package is installed (route A/B images, route C environments)
- Upstream: https://github.com/dauparas/ProteinMPNN
- Licence: MIT License
- Copyright line, exactly as printed in the upstream `LICENSE`: `Copyright (c) 2022 Justas Dauparas`
- Licence text in this tree: `third_party_licenses/ProteinMPNN.MIT.txt`, a byte-identical copy of the upstream `LICENSE`
- Kit files concerned: archive members only; the kit's design step does not load these parameters

## `template_dgram_head.npy` (inside ColabDesign, `colabdesign/af/weights/template_dgram_head.npy`)

- A member of `stock/colabdesign-e31a56fe.tar.gz` (and of every installed copy of the pinned package): one float32 array of shape
  (5, 128, 39), 99,968 bytes — per AlphaFold model, a linear head from the 128-channel pair representation to the 39 template-distogram bins.
- Provenance: ColabDesign's own file, added to the upstream repository by its author in commit `57849247c6` ("adding retrained distrogram
  head") together with the empty `colabdesign/af/weights/__init__.py`; upstream's example notebook `af/examples/af_pseudo_diffusion_dgram.ipynb`
  (not part of the archive) describes it as the AlphaFold distogram head "retrained to map output bins to template bins" and is the only
  upstream code that loads it. No module of the pinned package reads the file, and the kit does not read it.
- Terms: distributed by ColabDesign under its licence notice (section *ColabDesign* above; `third_party_licenses/ColabDesign.Beerware.txt`).
  It is not one of the AlphaFold parameter files and contains none of them.

## BindCraft

- Upstream: https://github.com/martinpacesa/BindCraft at commit `efb5bfeb8b4b1a5944256f979c34e0c8e6a82d9d` (the pin in `stock/PINS.json`)
- Licence: MIT License
- Copyright line, exactly as printed in the upstream `LICENSE`: `Copyright (c) 2024 Martin Pacesa`
- Licence text in this tree: `third_party_licenses/BindCraft.MIT.txt`, a byte-identical copy of the upstream `LICENSE` at that commit (the same
  bytes are carried at `stock/src/bindcraft/LICENSE` and inside `stock/bindcraft-efb5bfeb.tar.gz`)

| Kit file | Relation to the upstream code |
|---|---|
| `stock/bindcraft-efb5bfeb.tar.gz` | `git archive` of the pinned commit without the two prebuilt executables named below (recipe and omission in `stock/PINS.json`); every other member unmodified |
| `stock/src/bindcraft/**` | unmodified copies of the archive members that make up the design step (`functions/*.py`, settings files, `example/PDL1.pdb`, `LICENSE`) |
| `opt/colabdesign_opt/bindcraft.py`, `driver.py`, `settings.py` | import and call BindCraft's `binder_hallucination` and its settings files from `stock/src/bindcraft/` at run time; `pyrosetta_utils` is replaced by a module that refuses by name; no BindCraft source is copied into `opt/` |

Executables in BindCraft's repository: BindCraft distributes two prebuilt third-party programs, `functions/dssp` (DSSP, secondary-structure
assignment) and `functions/DAlphaBall.gcc`, whose licences its repository does not state. Neither is redistributed by this kit: the archive
under `stock/` and the copies under `stock/src/bindcraft/` leave both out (the omission is recorded with the archive recipe in
`stock/PINS.json`). The kit's design step needs `dssp`, which BindCraft's `functions/biopython_utils.py` runs: `bash run.sh install` fetches
it from the BindCraft repository at the pinned commit into `stock/src/bindcraft/functions/dssp` (listed in `.gitignore`) and checks it
against the digest pinned in `stock/PINS.json` before it is used; its licence terms remain those of its own authors.
`DAlphaBall.gcc` is used only by BindCraft stages this kit does not run and is neither distributed nor fetched.

## Data files

Outside `stock/` the kit carries no data file. The data-carrying members of the two upstream archives (and their unpacked copies under
`stock/src/`) are:

| File | What it holds | Source and terms |
|---|---|---|
| `stock/src/bindcraft/example/PDL1.pdb` (archive member `example/PDL1.pdb`; the README's example target) | BindCraft's example target: a 115-residue crop (chain A, residues 18-132, experimental B-factors kept, no header records) of a human PD-L1 crystal structure; the file names no wwPDB entry | distributed by BindCraft (MIT); deposited PD-L1 structures are wwPDB archive data (CC0 1.0) |
| `stock/src/bindcraft/settings_*/*.json` and the archive's other `settings_advanced/*.json`, `settings_filters/*.json`, `settings_target/PDL1.json` | BindCraft's design-stage and filter settings (parameters only) | BindCraft (MIT) |
| archive members `notebooks/BindCraft.ipynb`, `pipeline.png`, `bindcraft.slurm`, `install_bindcraft.sh` | BindCraft's notebook, pipeline figure and scripts, as released; not used by the kit | BindCraft (MIT) |
| archive members `colabdesign/mpnn/weights/*.pkl`, `colabdesign/mpnn/weights_soluble/*.pkl` (8 files) | ProteinMPNN parameter files as released by ColabDesign | section *ProteinMPNN parameters* above (MIT) |
| archive member `colabdesign/af/weights/template_dgram_head.npy` | see the section above | ColabDesign's licence notice |

No AlphaFold parameters, sequence databases or alignments are in this tree.

## Prebuilt binaries the kit binds (shared library, `common/opt_core/opt_core/kernels/triattn_xla/bin/**`)

Lever `txla` (mode `fast`) calls the shared library's XLA custom-call bridge for triangle attention, whose prebuilt binaries are described
member by member in `common/opt_core/opt_core/kernels/triattn_xla/NOTICE`. Third-party material those binaries were built against:

| Binary | Built against | Licence · copyright line as printed upstream | Text in this tree |
|---|---|---|---|
| `bin/cuda/sm_90a/libtriattn_m1_xla.so` | NVIDIA CUTLASS headers (release 4.7.1); its source `fa3_utils.h` carries portions adapted from FlashAttention-3 (`hopper/utils.h`, `hopper/softmax.h`) | CUTLASS: BSD-3-Clause, `Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` · FlashAttention: BSD-3-Clause, `Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.` (the adapted header itself reads `Copyright (c) 2024, Tri Dao.`) | `third_party_licenses/NVIDIA-CUTLASS.LICENSE.txt`, `third_party_licenses/Dao-AILab-flash-attention.LICENSE.txt` |
| `bin/cuda/sm_90a/*.so`, `bin/cuda/sm_80/libtriattn_sm80_xla.so` | NVIDIA CUDA Toolkit (nvcc; CUDA runtime linked statically) | NVIDIA CUDA Toolkit EULA (reference only; no text carried) | — |
| `bin/launcher/ffi-*/libtriattn_xla_launch.so` | jaxlib XLA FFI headers | Apache License, Version 2.0 (the JAX / XLA projects) | `third_party_licenses/AlphaFold.Apache-2.0.txt` is the same licence text |
| `bin/k2b/sm_*/*.cubin` | machine code produced by the Triton compiler and NVIDIA ptxas from the shared library's own kernel source | Triton: MIT License (compiler; no Triton source or binary is carried) | — |

The BSD-3-Clause texts above accompany the binary redistribution as their clause 2 requires; no CUTLASS or FlashAttention source file is
carried in this kit.
